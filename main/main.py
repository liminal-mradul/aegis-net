#!/usr/bin/env python3
"""
Aegis Distributed Medical Data Network
Main entry point for node initialization
"""

import argparse
import sys
from pathlib import Path

from src.node import AegisNode
from src.config import NodeConfig
from src.utils.logger import setup_logger

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Aegis Distributed Medical Data Network',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Create founder node
  python3 main.py --mode create --port 5000 --name Hospital_A --data-file hospital_a.csv
  
  # Join existing network
  python3 main.py --mode join --port 5001 --name Hospital_B --network <NETWORK_ID>
  
  # Create node on specific IP for LAN access
  python3 main.py --mode create --port 5000 --name Hospital_A --host 0.0.0.0
  
  # Verbose mode for debugging
  python3 main.py --mode create --port 5000 --name Hospital_A --verbose
        '''
    )
    
    # Required arguments
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['create', 'join'],
        help='Mode: create new network or join existing'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        required=True,
        help='Port number for this node (e.g., 5000)'
    )
    
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Node name (e.g., Hospital_A)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--network',
        type=str,
        help='Network ID to join (required for join mode)'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to CSV data file (uses synthetic data if not provided)'
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='0.0.0.0',
        help='Host address to bind (default: 0.0.0.0 for all interfaces)'
    )
    
    parser.add_argument(
        '--stake',
        type=float,
        default=100.0,
        help='Initial stake for voting (default: 100.0)'
    )
    
    parser.add_argument(
        '--reputation',
        type=float,
        default=0.5,
        help='Initial reputation (default: 0.5)'
    )
    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=10.0,
        help='Total privacy budget epsilon (default: 10.0)'
    )
    
    parser.add_argument(
        '--delta',
        type=float,
        default=1e-5,
        help='Privacy parameter delta (default: 1e-5)'
    )
    
    parser.add_argument(
        '--data-size',
        type=int,
        default=1000,
        help='Synthetic data size if no CSV (default: 1000)'
    )
    
    parser.add_argument(
        '--num-features',
        type=int,
        default=20,
        help='Number of features for synthetic data (default: 20)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Validation
    if args.mode == 'join' and not args.network:
        parser.error("--network is required when using --mode join")
    
    if args.data_file and not Path(args.data_file).exists():
        parser.error(f"Data file not found: {args.data_file}")
    
    if not (1024 <= args.port <= 65535):
        parser.error(f"Port must be between 1024 and 65535, got {args.port}")
    
    return args

def print_banner():
    """Print ASCII banner"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║     █████╗ ███████╗ ██████╗ ██╗███████╗                          ║
║    ██╔══██╗██╔════╝██╔════╝ ██║██╔════╝                          ║
║    ███████║█████╗  ██║  ███╗██║███████╗                          ║
║    ██╔══██║██╔══╝  ██║   ██║██║╚════██║                          ║
║    ██║  ██║███████╗╚██████╔╝██║███████║                          ║
║    ╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝                          ║
║                                                                   ║
║         Distributed Medical Data Network                          ║
║         Privacy-Preserving Federated Learning                     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def main():
    """Main entry point"""
    print_banner()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    logger = setup_logger('aegis.main', 'DEBUG' if args.verbose else 'INFO')
    logger.info(f"Initializing node: {args.name} on port {args.port}")
    
    try:
        # Create node configuration
        config = NodeConfig(
            port=args.port,
            host=args.host,
            node_name=args.name,
            stake=args.stake,
            reputation=args.reputation,
            epsilon_total=args.epsilon,
            delta_total=args.delta,
            data_file=args.data_file,
            data_size=args.data_size,
            num_features=args.num_features,
            verbose=args.verbose
        )
        
        # Initialize node
        node = AegisNode(config)
        
        # Create or join network
        if args.mode == 'create':
            network_id = node.create_network()
            logger.info("="*60)
            logger.info(f"NETWORK ID: {network_id}")
            logger.info("="*60)
            logger.info("Share this Network ID with other nodes to join")
        else:
            node.join_network(args.network)
            logger.info(f"Joined network: {args.network}")
        
        # Start interactive shell
        logger.info("Starting interactive shell (type 'help' for commands)")
        node.start_interactive_shell()
        
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()