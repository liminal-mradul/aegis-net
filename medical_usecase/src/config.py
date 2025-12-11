import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class NodeConfig:
    """Configuration for Aegis node"""
    
    # Network settings
    port: int = 5000
    host: str = '0.0.0.0'
    max_peers: int = 25
    
    # Consensus settings
    stake: float = 100.0
    reputation: float = 0.5
    consensus_timeout: int = 5
    vote_timeout: int = 3
    commit_timeout: int = 2
    
    # Privacy settings
    epsilon_total: float = 10.0
    delta_total: float = 1e-5
    dp_mechanism: str = 'gaussian'
    
    # Blockchain settings
    block_difficulty: int = 4
    blocks_per_checkpoint: int = 100
    
    # Data settings
    data_size: int = 1000
    num_features: int = 20
    data_file: Optional[str] = None
    
    # Node identity
    node_name: Optional[str] = None
    node_id: Optional[str] = None
    
    # System settings
    verbose: bool = False
    log_dir: str = './logs'
    shared_files_dir: str = './shared_files'
    max_file_size: int = 100 * 1024 * 1024  # 100 MB
    
    def __post_init__(self):
        """Validate configuration and create directories"""
        # Set default node name
        if not self.node_name:
            self.node_name = f"Node_{self.port}"
        
        # Validate parameters
        self._validate()
        
        # Create directories with restrictive permissions
        self._create_directories()
    
    def _validate(self):
        """Validate configuration parameters"""
        # Port validation
        if not (1024 <= self.port <= 65535):
            raise ValueError(f"Port must be in range 1024-65535, got {self.port}")
        
        # Stake validation
        if self.stake <= 0:
            raise ValueError(f"Stake must be positive, got {self.stake}")
        if self.stake > 1_000_000:
            raise ValueError(f"Stake exceeds maximum (1,000,000), got {self.stake}")
        
        # Reputation validation
        if not (0.0 <= self.reputation <= 1.0):
            raise ValueError(f"Reputation must be in [0, 1], got {self.reputation}")
        
        # Privacy parameter validation
        if self.epsilon_total <= 0:
            raise ValueError(f"Epsilon must be positive, got {self.epsilon_total}")
        if not (0 <= self.delta_total < 1):
            raise ValueError(f"Delta must be in [0, 1), got {self.delta_total}")
        
        # Timeout validation
        if self.consensus_timeout <= 0:
            raise ValueError(f"Consensus timeout must be positive")
        if self.vote_timeout <= 0:
            raise ValueError(f"Vote timeout must be positive")
        if self.commit_timeout <= 0:
            raise ValueError(f"Commit timeout must be positive")
        
        # Difficulty validation
        if not (1 <= self.block_difficulty <= 8):
            raise ValueError(f"Block difficulty must be in [1, 8], got {self.block_difficulty}")
        
        # Data validation
        if self.data_size <= 0:
            raise ValueError(f"Data size must be positive, got {self.data_size}")
        if self.num_features <= 0:
            raise ValueError(f"Number of features must be positive, got {self.num_features}")
        
        # File size validation
        if self.max_file_size <= 0:
            raise ValueError(f"Max file size must be positive, got {self.max_file_size}")
    
    def _create_directories(self):
        """Create required directories with secure permissions"""
        # Logs directory
        os.makedirs(self.log_dir, mode=0o700, exist_ok=True)
        
        # Shared files directory (contains medical data - must be secure)
        os.makedirs(self.shared_files_dir, mode=0o700, exist_ok=True)
        
        # Update permissions on existing directories
        try:
            os.chmod(self.log_dir, 0o700)
            os.chmod(self.shared_files_dir, 0o700)
        except Exception as e:
            # Log warning but don't fail (might not have permissions)
            pass

@dataclass
class NetworkConfig:
    """Configuration for federated network"""
    
    network_id: str
    genesis_timestamp: float
    founder_node: str
    
    # Consensus parameters
    min_nodes_for_consensus: int = 3
    byzantine_threshold: float = 0.33
    
    # Anomaly detection thresholds
    anomaly_zscore_threshold: float = 3.0
    anomaly_iqr_multiplier: float = 1.5
    
    def __post_init__(self):
        """Validate network configuration"""
        # Network ID validation
        if not isinstance(self.network_id, str) or len(self.network_id) != 32:
            raise ValueError(f"Invalid network ID format")
        
        # Byzantine threshold validation
        if not (0 <= self.byzantine_threshold < 0.34):
            raise ValueError(
                f"Byzantine threshold must be < 0.34, got {self.byzantine_threshold}"
            )
        
        # Minimum nodes validation
        if self.min_nodes_for_consensus < 1:
            raise ValueError(
                f"Min nodes must be >= 1, got {self.min_nodes_for_consensus}"
            )
        
        # Anomaly thresholds validation
        if self.anomaly_zscore_threshold <= 0:
            raise ValueError(f"Z-score threshold must be positive")
        if self.anomaly_iqr_multiplier <= 0:
            raise ValueError(f"IQR multiplier must be positive")
