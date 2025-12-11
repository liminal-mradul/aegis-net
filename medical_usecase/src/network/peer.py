import socket
import threading
import time
import logging
from typing import Dict, Callable, Optional
import requests
from flask import Flask, request, jsonify

from ..utils.logger import setup_logger
from .protocol import Message, MessageType

class PeerManager:
    def __init__(self, node_id: str, port: int, host: str = '0.0.0.0'):
        self.node_id = node_id
        self.port = port
        self.host = host
        self.peers: Dict[str, Dict] = {}
        self.message_handlers: Dict[str, Callable] = {}
        self.logger = setup_logger(f'aegis.peer.{port}')
        
        # Create Flask app with logging suppressed
        self.app = Flask(__name__)
        self._setup_routes()
        
        # Suppress Flask logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        self.app.logger.setLevel(logging.ERROR)
        
        self.server_thread = None
        self.running = False
        
        # Store voting manager reference (set later by node)
        self.voting_manager = None
        
        # Register heartbeat handler
        self.register_handler(MessageType.HEARTBEAT.value, self.handle_heartbeat)
        
        self.logger.info(f"Peer manager initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup all Flask routes"""
        self.app.add_url_rule('/message', 'receive_message', 
                             self.receive_message, methods=['POST'])
        self.app.add_url_rule('/health', 'health_check',
                             self.health_check, methods=['GET'])
        self.app.add_url_rule('/register_peer', 'register_peer',
                             self.register_peer_endpoint, methods=['POST'])
        self.app.add_url_rule('/info', 'node_info',
                             self.node_info, methods=['GET'])
    
    def start(self):
        """Start peer manager server"""
        self.running = True
        self.server_thread = threading.Thread(
            target=self._run_server,
            daemon=True
        )
        self.server_thread.start()
        self.logger.info(f"✓ Server started on {self.host}:{self.port}")
        
        # Start heartbeat thread
        heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True
        )
        heartbeat_thread.start()
    
    def stop(self):
        """Stop peer manager"""
        self.running = False
        self.logger.info("Peer manager shutting down")
    
    def _run_server(self):
        """Run Flask server with output suppressed"""
        import sys
        
        # Suppress Flask startup messages
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        
        self.app.run(
            host=self.host, 
            port=self.port, 
            threaded=True, 
            debug=False,
            use_reloader=False
        )
    
    def register_handler(self, msg_type: str, handler: Callable):
        """Register message handler"""
        self.message_handlers[msg_type] = handler
        self.logger.debug(f"Registered handler for {msg_type}")
    
    def add_peer(self, peer_id: str, peer_url: str):
        """Add peer to network"""
        if peer_id == self.node_id:
            return
        
        self.peers[peer_id] = {
            'url': peer_url,
            'last_seen': time.time(),
            'reputation': 0.5,
            'failed_attempts': 0
        }
        self.logger.info(f"✓ Added peer {peer_id[:8]}... at {peer_url}")
    
    def remove_peer(self, peer_id: str):
        """Remove peer from network"""
        if peer_id in self.peers:
            del self.peers[peer_id]
            self.logger.info(f"✗ Removed peer {peer_id[:8]}...")
    
    def send_message(self, peer_id: str, message: Message) -> bool:
        """Send message to specific peer"""
        if peer_id not in self.peers:
            self.logger.warning(f"Unknown peer {peer_id[:8]}...")
            return False
        
        peer = self.peers[peer_id]
        try:
            response = requests.post(
                f"{peer['url']}/message",
                json=message.to_json(),
                timeout=5
            )
            
            if response.status_code == 200:
                peer['last_seen'] = time.time()
                peer['failed_attempts'] = 0
                return True
            else:
                self.logger.warning(
                    f"Failed to send to {peer_id[:8]}: HTTP {response.status_code}"
                )
                peer['failed_attempts'] += 1
                return False
                
        except Exception as e:
            self.logger.debug(f"Error sending to {peer_id[:8]}: {e}")
            peer['failed_attempts'] += 1
            
            if peer['failed_attempts'] >= 5:
                self.logger.warning(f"Removing unresponsive peer {peer_id[:8]}...")
                self.remove_peer(peer_id)
            
            return False
    
    def broadcast_message(self, message: Message) -> int:
        """Broadcast message to all peers"""
        success_count = 0
        for peer_id in list(self.peers.keys()):
            if self.send_message(peer_id, message):
                success_count += 1
        
        self.logger.debug(f"Broadcast: {success_count}/{len(self.peers)} peers reached")
        return success_count
    
    def receive_message(self):
        """Handle incoming message"""
        try:
            json_data = request.get_json()
            message = Message.from_json(json_data)
            
            # Update last seen time
            if message.sender_id in self.peers:
                self.peers[message.sender_id]['last_seen'] = time.time()
            
            # Dispatch to handler
            handler = self.message_handlers.get(message.msg_type)
            if handler:
                threading.Thread(
                    target=handler,
                    args=(message,),
                    daemon=True
                ).start()
            else:
                # Only log if not heartbeat (reduce noise)
                if message.msg_type != MessageType.HEARTBEAT.value:
                    self.logger.warning(f"No handler for {message.msg_type}")
            
            return jsonify({'status': 'ok'}), 200
            
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    def health_check(self):
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'node_id': self.node_id,
            'peers': len(self.peers),
            'active_peers': len(self.get_active_peers())
        }), 200
    
    def register_peer_endpoint(self):
        """Handle incoming peer registration - NEW for bidirectional sync"""
        try:
            data = request.get_json()
            peer_id = data['node_id']
            peer_url = data['url']
            stake = data.get('stake', 100.0)
            reputation = data.get('reputation', 0.5)
            
            # Add to our peer list if not already there
            if peer_id not in self.peers:
                self.add_peer(peer_id, peer_url)
                
                # Also register in voting manager if available
                if self.voting_manager:
                    self.voting_manager.register_node(peer_id, stake, reputation)
                    self.logger.debug(f"Registered {peer_id[:8]} in voting system")
            
            return jsonify({
                'status': 'registered',
                'node_id': self.node_id,
                'message': 'Peer registered successfully'
            }), 200
            
        except Exception as e:
            self.logger.error(f"Error in peer registration: {e}", exc_info=True)
            return jsonify({'status': 'error', 'message': str(e)}), 400
    
    def node_info(self):
        """Return node information - NEW endpoint for discovery"""
        return jsonify({
            'node_id': self.node_id,
            'peers': [
                {
                    'id': pid[:16] + '...',
                    'url': self.peers[pid]['url'],
                    'last_seen': time.time() - self.peers[pid]['last_seen'],
                    'reputation': self.peers[pid]['reputation']
                }
                for pid in self.peers.keys()
            ],
            'active_peers': len(self.get_active_peers())
        }), 200
    
    def handle_heartbeat(self, message: Message):
        """Handle heartbeat message"""
        # Heartbeat already handled by updating last_seen in receive_message
        pass
    
    def _heartbeat_loop(self):
        """Send periodic heartbeats to all peers"""
        while self.running:
            time.sleep(10)
            
            current_time = time.time()
            for peer_id in list(self.peers.keys()):
                peer = self.peers[peer_id]
                
                # Check if peer is offline
                if current_time - peer['last_seen'] > 60:
                    self.logger.warning(f"Peer {peer_id[:8]} appears offline")
                    peer['reputation'] = max(0, peer['reputation'] - 0.1)
                
                # Send heartbeat
                message = Message(
                    msg_type=MessageType.HEARTBEAT.value,
                    sender_id=self.node_id,
                    data={'timestamp': current_time},
                    timestamp=current_time
                )
                self.send_message(peer_id, message)
    
    def get_active_peers(self) -> list:
        """Get list of active peers (seen in last 60s)"""
        current_time = time.time()
        return [
            peer_id for peer_id, peer in self.peers.items()
            if current_time - peer['last_seen'] < 60
        ]
    
    def get_local_ip(self) -> str:
        """Get local IP address for LAN connections"""
        try:
            # Create a socket to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except Exception:
            return "ISSUE: NO CONNECTION"