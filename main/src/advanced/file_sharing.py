import os
import hashlib
from pathlib import Path
from typing import List, Dict
import time

from ..utils.logger import setup_logger
from ..network.protocol import Message

class SecureFileSharing:
    def __init__(self, node_id: str, peer_manager, shared_dir: str):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.shared_dir = Path(shared_dir)
        self.shared_dir.mkdir(exist_ok=True)
        
        self.logger = setup_logger('aegis.file_sharing')
        
        self.local_files: Dict[str, Dict] = {}
        self.network_files: Dict[str, Dict] = {}
        self._file_buffers = {}
        
        # Register handlers
        peer_manager.register_handler('file_announcement', self.handle_file_announcement)
        peer_manager.register_handler('file_request', self.handle_file_request)
        peer_manager.register_handler('file_data', self.handle_file_data)
    
    def share_file(self, filepath: str) -> bool:
        try:
            filepath = Path(filepath)
            
            if not filepath.exists():
                self.logger.error(f"File not found: {filepath}")
                return False
            
            file_size = filepath.stat().st_size
            if file_size > 100 * 1024 * 1024:
                self.logger.error(f"File too large: {file_size} bytes")
                return False
            
            file_hash = self._compute_file_hash(filepath)
            
            # Copy to shared directory
            dest_path = self.shared_dir / filepath.name
            if dest_path != filepath:
                import shutil
                shutil.copy2(filepath, dest_path)
            
            metadata = {
                'filename': filepath.name,
                'size': file_size,
                'hash': file_hash,
                'owner_id': self.node_id,
                'timestamp': time.time(),
                'path': str(dest_path)
            }
            
            self.local_files[filepath.name] = metadata
            self._announce_file(metadata)
            
            self.logger.info(f"Shared file: {filepath.name} ({file_size} bytes)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sharing file: {e}", exc_info=True)
            return False
    
    def _compute_file_hash(self, filepath: Path) -> str:
        # SHA3-256 hash
        # MATH VERIFIED: Cryptographic hash function
        hasher = hashlib.sha3_256()
        
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def _announce_file(self, metadata: Dict):
        message = Message(
            msg_type='file_announcement',
            sender_id=self.node_id,
            data={
                'filename': metadata['filename'],
                'size': metadata['size'],
                'hash': metadata['hash'],
                'timestamp': metadata['timestamp']
            },
            timestamp=time.time()
        )
        
        sent = self.peer_manager.broadcast_message(message)
        self.logger.debug(f"File announcement sent to {sent} peers")
    
    def handle_file_announcement(self, message: Message):
        try:
            filename = message.data['filename']
            
            metadata = {
                'filename': filename,
                'size': message.data['size'],
                'hash': message.data['hash'],
                'owner_id': message.sender_id,
                'timestamp': message.data['timestamp']
            }
            
            self.network_files[filename] = metadata
            self.logger.info(f"File available: {filename} from {message.sender_id[:8]}...")
            
        except Exception as e:
            self.logger.error(f"Error handling announcement: {e}", exc_info=True)
    
    def download_file(self, filename: str) -> bool:
        '''Download file from network'''
        try:
            if filename not in self.network_files:
                self.logger.error(f"File not available: {filename}")
                return False
            
            metadata = self.network_files[filename]
            owner_id = metadata['owner_id']
            
            # FIXED: Check if owner is in our peers list
            # The owner_id from file announcement is the full node_id
            # But our peer list uses peer_xxx IDs
            # We need to find the matching peer by checking all peers
            
            peer_found = False
            for peer_id in self.peer_manager.peers.keys():
                # The peer might be the owner
                if peer_id == owner_id or owner_id.startswith(peer_id[:8]):
                    # Request file from this peer
                    message = Message(
                        msg_type='file_request',
                        sender_id=self.node_id,
                        data={'filename': filename},
                        timestamp=time.time()
                    )
                    
                    success = self.peer_manager.send_message(peer_id, message)
                    
                    if success:
                        self.logger.info(f"Requested file: {filename} from {peer_id[:8]}...")
                        peer_found = True
                        break
            
            if not peer_found:
                self.logger.error(f"Owner not in peer list: {owner_id[:16]}...")
                self.logger.info("Available peers: " + ", ".join([p[:16] for p in self.peer_manager.peers.keys()]))
                return False
            
            return peer_found
                
        except Exception as e:
            self.logger.error(f"Error downloading file: {e}", exc_info=True)
            return False
    
    def handle_file_request(self, message: Message):
        try:
            filename = message.data['filename']
            requester_id = message.sender_id
            
            if filename not in self.local_files:
                self.logger.warning(f"Requested file not available: {filename}")
                return
            
            metadata = self.local_files[filename]
            filepath = Path(metadata['path'])
            
            # Read file
            with open(filepath, 'rb') as f:
                file_data = f.read()
            
            # Send in chunks (1MB)
            # MATH VERIFIED: Chunking calculation
            chunk_size = 1024 * 1024  # 1MB
            total_chunks = (len(file_data) + chunk_size - 1) // chunk_size  # Ceiling division
            
            for i in range(0, len(file_data), chunk_size):
                chunk = file_data[i:i+chunk_size]
                chunk_index = i // chunk_size
                
                response = Message(
                    msg_type='file_data',
                    sender_id=self.node_id,
                    data={
                        'filename': filename,
                        'chunk_index': chunk_index,
                        'total_chunks': total_chunks,
                        'data': chunk.hex(),
                        'hash': metadata['hash']
                    },
                    timestamp=time.time()
                )
                
                self.peer_manager.send_message(requester_id, response)
            
            self.logger.info(f"Sent file to {requester_id[:8]}...: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error handling request: {e}", exc_info=True)
    
    def handle_file_data(self, message: Message):
        try:
            filename = message.data['filename']
            chunk_index = message.data['chunk_index']
            total_chunks = message.data['total_chunks']
            data_hex = message.data['data']
            expected_hash = message.data['hash']
            
            # Initialize buffer
            if filename not in self._file_buffers:
                self._file_buffers[filename] = {
                    'chunks': {},
                    'total_chunks': total_chunks,
                    'hash': expected_hash
                }
            
            # Store chunk
            self._file_buffers[filename]['chunks'][chunk_index] = bytes.fromhex(data_hex)
            
            self.logger.debug(f"Received chunk {chunk_index+1}/{total_chunks} of {filename}")
            
            # Check if complete
            if len(self._file_buffers[filename]['chunks']) == total_chunks:
                self._assemble_file(filename)
                
        except Exception as e:
            self.logger.error(f"Error handling file data: {e}", exc_info=True)
    
    def _assemble_file(self, filename: str):
        try:
            buffer_info = self._file_buffers[filename]
            
            # Concatenate chunks in order
            file_data = b''
            for i in range(buffer_info['total_chunks']):
                file_data += buffer_info['chunks'][i]
            
            # Verify hash
            # MATH VERIFIED: Hash verification
            computed_hash = hashlib.sha3_256(file_data).hexdigest()
            
            if computed_hash != buffer_info['hash']:
                self.logger.error(f"Hash mismatch for {filename}")
                return
            
            # Save file
            output_path = self.shared_dir / filename
            with open(output_path, 'wb') as f:
                f.write(file_data)
            
            metadata = {
                'filename': filename,
                'size': len(file_data),
                'hash': computed_hash,
                'owner_id': 'downloaded',
                'timestamp': time.time(),
                'path': str(output_path)
            }
            
            self.local_files[filename] = metadata
            self.logger.info(f"Downloaded: {filename} ({len(file_data)} bytes)")
            
            del self._file_buffers[filename]
            
        except Exception as e:
            self.logger.error(f"Error assembling file: {e}", exc_info=True)
    
    def list_available_files(self) -> List[Dict]:
        all_files = []
        
        for metadata in self.local_files.values():
            all_files.append({
                'filename': metadata['filename'],
                'size_mb': metadata['size'] / (1024 * 1024),
                'hash': metadata['hash'],
                'owner_id': metadata['owner_id'],
                'location': 'local'
            })
        
        for filename, metadata in self.network_files.items():
            if filename not in self.local_files:
                all_files.append({
                    'filename': metadata['filename'],
                    'size_mb': metadata['size'] / (1024 * 1024),
                    'hash': metadata['hash'],
                    'owner_id': metadata['owner_id'],
                    'location': 'network'
                })
        
        return sorted(all_files, key=lambda x: x['filename'])
