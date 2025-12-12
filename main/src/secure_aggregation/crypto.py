import numpy as np
import secrets
import hashlib
from typing import Tuple

class SecretSharing:
    """
    Shamir secret sharing over finite field.
    Enables (t, n) threshold secret reconstruction.
    """
    
    # Mersenne prime for finite field arithmetic
    PRIME = 2**127 - 1
    
    @classmethod
    def share_secret(cls, secret: int, threshold: int, n_shares: int) -> list:
        """
        Generate (threshold, n_shares) Shamir secret shares.
        
        Args:
            secret: Secret value to share
            threshold: Minimum shares needed for reconstruction
            n_shares: Total number of shares to generate
        
        Returns:
            List of (x, y) share tuples
        
        Raises:
            ValueError: If threshold > n_shares
        """
        if threshold > n_shares:
            raise ValueError(
                f"Threshold ({threshold}) cannot exceed number of shares ({n_shares})"
            )
        if threshold < 1:
            raise ValueError(f"Threshold must be >= 1, got {threshold}")
        if n_shares < 1:
            raise ValueError(f"Number of shares must be >= 1, got {n_shares}")
        
        # Generate random polynomial coefficients
        # P(x) = secret + a1*x + a2*x^2 + ... + a_{t-1}*x^{t-1}
        coefficients = [secret] + [
            secrets.randbelow(cls.PRIME) for _ in range(threshold - 1)
        ]
        
        # Evaluate polynomial at points 1, 2, ..., n_shares
        shares = []
        for i in range(1, n_shares + 1):
            share = cls._eval_polynomial(coefficients, i)
            shares.append((i, share))
        
        return shares
    
    @classmethod
    def _eval_polynomial(cls, coefficients: list, x: int) -> int:
        """
        Evaluate polynomial at point x using Horner's method.
        All arithmetic is done modulo PRIME.
        """
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % cls.PRIME
        return result
    
    @classmethod
    def reconstruct_secret(cls, shares: list) -> int:
        """
        Reconstruct secret from shares using Lagrange interpolation.
        
        Args:
            shares: List of (x, y) share tuples (at least threshold shares)
        
        Returns:
            Reconstructed secret
        
        Raises:
            ValueError: If shares list is empty
        """
        if not shares:
            raise ValueError("Need at least one share for reconstruction")
        
        secret = 0
        
        # Lagrange interpolation at x=0
        for i, (x_i, y_i) in enumerate(shares):
            numerator = 1
            denominator = 1
            
            for j, (x_j, _) in enumerate(shares):
                if i != j:
                    # L_i(0) = Π_{j≠i} (0 - x_j) / (x_i - x_j)
                    #        = Π_{j≠i} (-x_j) / (x_i - x_j)
                    numerator = (numerator * (-x_j)) % cls.PRIME
                    denominator = (denominator * (x_i - x_j)) % cls.PRIME
            
            # Modular inverse using Fermat's little theorem
            # a^(-1) = a^(p-2) mod p for prime p
            lagrange_coeff = (numerator * pow(denominator, cls.PRIME - 2, cls.PRIME)) % cls.PRIME
            secret = (secret + y_i * lagrange_coeff) % cls.PRIME
        
        return secret

class PairwiseMasking:
    """
    Additive masking with pairwise secrets for secure aggregation.
    Masks cancel out when summed across all parties.
    """
    
    @staticmethod
    def generate_mask_seed(shared_secret: bytes, session_id: str, 
                          nonce: bytes = None) -> bytes:
        """
        Derive deterministic mask seed from shared secret.
        
        Args:
            shared_secret: Pre-shared secret between two parties
            session_id: Unique session identifier
            nonce: Optional additional randomness
        
        Returns:
            Derived seed for mask generation
        """
        if nonce is None:
            nonce = b''
        
        # Combine inputs with nonce for additional entropy
        combined = shared_secret + session_id.encode() + nonce
        return hashlib.sha3_256(combined).digest()
    
    @staticmethod
    def generate_mask(seed: bytes, dimension: int) -> np.ndarray:
        """
        Generate pseudorandom mask vector from seed.
        
        Args:
            seed: Random seed (from generate_mask_seed)
            dimension: Vector dimension
        
        Returns:
            Pseudorandom mask vector (int64)
        """
        if dimension <= 0:
            raise ValueError(f"Dimension must be positive, got {dimension}")
        
        # Use seed to initialize numpy random state
        seed_int = int.from_bytes(seed[:8], 'big')
        rng = np.random.RandomState(seed_int)
        
        # Generate mask in safe range to prevent overflow
        # Use ±2^30 range (well within int64 bounds)
        mask = rng.randint(
            -(2**30), 2**30, size=dimension, dtype=np.int64
        )
        return mask
    
    @staticmethod
    def apply_pairwise_masks(value: np.ndarray, 
                            shared_secrets: dict,
                            session_id: str,
                            node_id: str) -> np.ndarray:
        """
        Apply pairwise masks to value for secure aggregation.
        
        Masks are added for peers with greater IDs and subtracted for
        peers with smaller IDs. This ensures masks cancel when summed.
        
        Args:
            value: Vector to mask (int64)
            shared_secrets: Dict of peer_id -> shared_secret
            session_id: Session identifier
            node_id: This node's ID
        
        Returns:
            Masked vector
        """
        masked_value = value.copy()
        dimension = len(value)
        
        for peer_id, shared_secret in shared_secrets.items():
            # Generate deterministic mask
            seed = PairwiseMasking.generate_mask_seed(shared_secret, session_id)
            mask = PairwiseMasking.generate_mask(seed, dimension)
            
            # Add or subtract mask based on ID ordering
            if peer_id > node_id:
                masked_value += mask
            else:
                masked_value -= mask
        
        return masked_value
