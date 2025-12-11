import hashlib
import secrets
from typing import Tuple
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.backends import default_backend

class CryptoHelper:
    # Configurable curve - can be upgraded if needed
    CURVE = ec.SECP384R1()
    
    @staticmethod
    def generate_keypair() -> Tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]:
        """Generate ECDSA keypair over NIST P-384 curve"""
        private_key = ec.generate_private_key(CryptoHelper.CURVE, default_backend())
        public_key = private_key.public_key()
        return private_key, public_key
    
    @staticmethod
    def sign_data(private_key: ec.EllipticCurvePrivateKey, data: bytes) -> bytes:
        """Sign data with ECDSA"""
        signature = private_key.sign(
            data,
            ec.ECDSA(hashes.SHA256())
        )
        return signature
    
    @staticmethod
    def verify_signature(public_key: ec.EllipticCurvePublicKey, 
                        data: bytes, signature: bytes) -> bool:
        """Verify ECDSA signature"""
        try:
            public_key.verify(
                signature,
                data,
                ec.ECDSA(hashes.SHA256())
            )
            return True
        except Exception:
            return False
    
    @staticmethod
    def hash_data(data: bytes) -> str:
        """SHA3-256 hash"""
        return hashlib.sha3_256(data).hexdigest()
    
    @staticmethod
    def generate_node_id() -> str:
        """Generate 256-bit node identifier"""
        return secrets.token_hex(32)
    
    @staticmethod
    def generate_network_id() -> str:
        """Generate unique network identifier"""
        return secrets.token_hex(16)
    
    @staticmethod
    def secure_random_bytes(n: int) -> bytes:
        """Generate cryptographically secure random bytes"""
        return secrets.token_bytes(n)
