from typing import Any, Dict

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class Validators:
    # Maximum stake to prevent voting power concentration
    MAX_STAKE = 1_000_000.0
    
    @staticmethod
    def validate_port(port: int) -> None:
        """Validate port number is in valid range"""
        if not isinstance(port, int):
            raise ValidationError(f"Port must be integer, got {type(port).__name__}")
        if not (1024 <= port <= 65535):
            raise ValidationError(f"Port must be in range 1024-65535, got {port}")
    
    @staticmethod
    def validate_stake(stake: float) -> None:
        """Validate stake is positive and within reasonable bounds"""
        if not isinstance(stake, (int, float)):
            raise ValidationError(f"Stake must be numeric, got {type(stake).__name__}")
        if stake <= 0:
            raise ValidationError(f"Stake must be positive, got {stake}")
        if stake > Validators.MAX_STAKE:
            raise ValidationError(
                f"Stake exceeds maximum ({Validators.MAX_STAKE}), got {stake}"
            )
    
    @staticmethod
    def validate_reputation(reputation: float) -> None:
        """Validate reputation is in [0, 1] range"""
        if not isinstance(reputation, (int, float)):
            raise ValidationError(f"Reputation must be numeric, got {type(reputation).__name__}")
        if not (0.0 <= reputation <= 1.0):
            raise ValidationError(f"Reputation must be in [0, 1], got {reputation}")
    
    @staticmethod
    def validate_network_id(network_id: str) -> None:
        """Validate network ID format"""
        if not isinstance(network_id, str):
            raise ValidationError(f"Network ID must be string, got {type(network_id).__name__}")
        if len(network_id) != 32:
            raise ValidationError(f"Network ID must be 32 hex chars, got {len(network_id)}")
        try:
            int(network_id, 16)
        except ValueError:
            raise ValidationError("Network ID must be valid hexadecimal")
    
    @staticmethod
    def validate_node_id(node_id: str) -> None:
        """Validate node ID format"""
        if not isinstance(node_id, str):
            raise ValidationError(f"Node ID must be string, got {type(node_id).__name__}")
        if len(node_id) != 64:
            raise ValidationError(f"Node ID must be 64 hex chars, got {len(node_id)}")
        try:
            int(node_id, 16)
        except ValueError:
            raise ValidationError("Node ID must be valid hexadecimal")
    
    @staticmethod
    def validate_consensus_params(threshold: float, byzantine_ratio: float) -> None:
        """Validate consensus parameters"""
        if not (0 < threshold <= 1):
            raise ValidationError(f"Threshold must be in (0,1], got {threshold}")
        if not (0 <= byzantine_ratio < 0.34):
            raise ValidationError(f"Byzantine ratio must be < 0.34, got {byzantine_ratio}")
    
    @staticmethod
    def validate_privacy_params(epsilon: float, delta: float) -> None:
        """Validate differential privacy parameters"""
        if not isinstance(epsilon, (int, float)):
            raise ValidationError(f"Epsilon must be numeric, got {type(epsilon).__name__}")
        if epsilon <= 0:
            raise ValidationError(f"Epsilon must be positive, got {epsilon}")
        if not isinstance(delta, (int, float)):
            raise ValidationError(f"Delta must be numeric, got {type(delta).__name__}")
        if not (0 <= delta < 1):
            raise ValidationError(f"Delta must be in [0, 1), got {delta}")
