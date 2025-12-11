import numpy as np
import threading
import time
from typing import Dict, List, Optional

from ..utils.logger import setup_logger, OperationLogger
from ..utils.crypto_utils import CryptoHelper
from ..network.protocol import Message, MessageType
from .crypto import SecretSharing, PairwiseMasking

class SecureAggregator:
    def __init__(self, node_id: str, peer_manager):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.logger = setup_logger('aegis.aggregation')
        
        # Pairwise shared secrets
        self.shared_secrets: Dict[str, bytes] = {}
        
        # Aggregation session state
        self.session_id: Optional[str] = None
        self.contributions: Dict[str, np.ndarray] = {}
        self.dimension: Optional[int] = None
        self.threshold: int = 0
        self.lock = threading.Lock()
        
        # Initialize private key as None (set during key exchange)
        self._private_key = None
        
        # Register handlers
        peer_manager.register_handler(
            MessageType.KEY_EXCHANGE.value,
            self.handle_key_exchange
        )
        peer_manager.register_handler(
            MessageType.MASKED_CONTRIBUTION.value,
            self.handle_contribution
        )
    
    def establish_pairwise_keys(self) -> bool:
        """Establish shared secrets with all peers via ECDH - FIXED"""
        with OperationLogger(self.logger, "Pairwise key exchange"):
            # Clear old secrets
            self.shared_secrets.clear()
            
            # Generate ephemeral DH key
            private_key, public_key = CryptoHelper.generate_keypair()
            
            # Store for later use in handle_key_exchange
            self._private_key = private_key
            
            # Serialize public key
            from cryptography.hazmat.primitives import serialization
            public_bytes = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            )
            
            # Broadcast public key
            message = Message(
                msg_type=MessageType.KEY_EXCHANGE.value,
                sender_id=self.node_id,
                data={'public_key': public_bytes.hex()},
                timestamp=time.time()
            )
            
            num_peers = len(self.peer_manager.get_active_peers())
            if num_peers > 0:
                sent = self.peer_manager.broadcast_message(message)
                self.logger.debug(f"Key exchange broadcast to {sent} peers")
                
                # Wait longer for key exchange to complete
                time.sleep(5)
                
                num_secrets = len(self.shared_secrets)
                self.logger.info(
                    f"Key exchange complete: {num_secrets}/{num_peers} shared secrets established"
                )
                
                if num_secrets < num_peers:
                    self.logger.warning(
                        f"Failed to establish keys with {num_peers - num_secrets} peers. "
                        "Check network connectivity."
                    )
                
                return num_secrets > 0
            else:
                self.logger.warning("No peers available for key exchange")
                return False
    
    def handle_key_exchange(self, message: Message):
        """Receive peer's public key and compute shared secret - FIXED"""
        try:
            peer_id = message.sender_id
            public_key_hex = message.data['public_key']
            
            # Check if we have our private key yet
            if self._private_key is None:
                # Store for later processing
                self.logger.debug(f"Received key from {peer_id[:8]} before our key generated, will retry")
                # Wait a bit and try again
                time.sleep(0.5)
                if self._private_key is None:
                    self.logger.warning(f"Still no private key, skipping {peer_id[:8]}")
                    return
            
            # Deserialize peer's public key
            from cryptography.hazmat.primitives import serialization
            public_bytes = bytes.fromhex(public_key_hex)
            peer_public_key = serialization.load_pem_public_key(public_bytes)
            
            # Compute shared secret via ECDH
            from cryptography.hazmat.primitives.asymmetric import ec
            shared_key = self._private_key.exchange(
                ec.ECDH(),
                peer_public_key
            )
            
            with self.lock:
                self.shared_secrets[peer_id] = shared_key
            
            self.logger.debug(f"✓ Established shared secret with {peer_id[:8]}")
            
        except Exception as e:
            self.logger.error(f"Error in key exchange with {message.sender_id[:8]}: {e}", exc_info=True)
    
    def contribute(self, vector: np.ndarray, session_id: str) -> bool:
        """Contribute masked vector to aggregation - FIXED"""
        with OperationLogger(self.logger, "Secure contribution"):
            self.session_id = session_id
            self.dimension = len(vector)
            
            # Scale to fixed-point (4 decimal places)
            scaled_vector = (vector * 10000).astype(np.int64)
            
            # Check if we have shared secrets
            peer_ids = sorted(self.peer_manager.get_active_peers())
            num_secrets = len(self.shared_secrets)
            
            if num_secrets == 0 and len(peer_ids) > 0:
                self.logger.warning(
                    f"No shared secrets established - using unmasked values"
                )
                masked_vector = scaled_vector.copy()
            else:
                # Apply pairwise masks
                masked_vector = scaled_vector.copy()
                
                for peer_id in peer_ids:
                    if peer_id not in self.shared_secrets:
                        self.logger.debug(f"No shared secret with {peer_id[:8]}, skipping mask")
                        continue
                    
                    # Generate mask
                    seed = PairwiseMasking.generate_mask_seed(
                        self.shared_secrets[peer_id],
                        session_id
                    )
                    mask = PairwiseMasking.generate_mask(seed, self.dimension)
                    
                    # Add or subtract mask based on ordering
                    if peer_id > self.node_id:
                        masked_vector += mask
                    else:
                        masked_vector -= mask
            
            # Store own contribution
            with self.lock:
                self.contributions[self.node_id] = masked_vector
            
            # Broadcast masked contribution
            message = Message(
                msg_type=MessageType.MASKED_CONTRIBUTION.value,
                sender_id=self.node_id,
                data={
                    'session_id': session_id,
                    'contribution': masked_vector.tolist(),
                    'commitment': CryptoHelper.hash_data(masked_vector.tobytes())
                },
                timestamp=time.time()
            )
            
            sent = self.peer_manager.broadcast_message(message)
            self.logger.debug(f"Masked contribution sent to {sent} peers")
            
            return True
    
    def handle_contribution(self, message: Message):
        """Receive masked contribution from peer"""
        try:
            session_id = message.data['session_id']
            contribution = np.array(message.data['contribution'], dtype=np.int64)
            commitment = message.data['commitment']
            peer_id = message.sender_id
            
            # Verify commitment
            computed_commitment = CryptoHelper.hash_data(contribution.tobytes())
            if computed_commitment != commitment:
                self.logger.warning(f"Invalid commitment from {peer_id[:8]}")
                return
            
            with self.lock:
                if session_id != self.session_id:
                    self.logger.debug(
                        f"Ignoring contribution for different session: {session_id[:8]}"
                    )
                    return
                
                self.contributions[peer_id] = contribution
                self.logger.debug(
                    f"✓ Received contribution from {peer_id[:8]} "
                    f"({len(self.contributions)} total)"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling contribution: {e}", exc_info=True)
    
    def aggregate(self, timeout: float = 10.0) -> Optional[np.ndarray]:
        """Aggregate masked contributions - FIXED"""
        with OperationLogger(self.logger, "Aggregation"):
            # Wait for contributions
            deadline = time.time() + timeout
            num_expected = len(self.peer_manager.peers) + 1
            
            while time.time() < deadline:
                with self.lock:
                    num_contributions = len(self.contributions)
                
                if num_contributions >= num_expected:
                    break
                
                time.sleep(0.5)
            
            with self.lock:
                if not self.contributions:
                    self.logger.error("No contributions received")
                    return None
                
                num_contributions = len(self.contributions)
                
                if num_contributions < num_expected:
                    self.logger.warning(
                        f"Only received {num_contributions}/{num_expected} contributions. "
                        "Proceeding with available data."
                    )
                
                # Sum all masked contributions
                # Masks cancel out due to pairwise +/- structure
                aggregate = np.sum(
                    list(self.contributions.values()),
                    axis=0
                )
                
                # Convert back to float
                result = aggregate / 10000.0
                
                self.logger.info(
                    f"✓ Aggregation complete: {num_contributions} contributors, "
                    f"dimension={len(result)}"
                )
                
                # Clear session state
                self.contributions.clear()
                self.session_id = None
                
                return result
    
    def get_aggregation_status(self) -> Dict:
        """Return current aggregation status"""
        with self.lock:
            return {
                'session_id': self.session_id,
                'dimension': self.dimension,
                'contributions_received': len(self.contributions),
                'shared_secrets_established': len(self.shared_secrets)
            }