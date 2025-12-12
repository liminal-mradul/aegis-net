import time
import uuid
import json
from typing import Optional
import numpy as np
from pathlib import Path

from .config import NodeConfig, NetworkConfig
from .utils.logger import setup_logger, OperationLogger
from .utils.crypto_utils import CryptoHelper
from .utils.validators import Validators
from .network.peer import PeerManager
from .consensus.voting import VotingManager
from .consensus.protocol import ConsensusProtocol
from .secure_aggregation.aggregator import SecureAggregator
from .privacy.budget import PrivacyBudgetManager
from .privacy.dp_engine import DifferentialPrivacyEngine
from .audit.blockchain import AuditBlockchain
from .medical.data_generator import MedicalDataGenerator
from .medical.model import DiseasePredictionModel

# Import advanced features
from .advanced.csv_loader import CSVMedicalDataLoader
from .advanced.anomaly_detector import AnomalyDetector
from .advanced.file_sharing import SecureFileSharing

class AegisNode:
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = setup_logger('aegis.node', 'DEBUG' if config.verbose else 'INFO')
        
        if not config.node_id:
            config.node_id = CryptoHelper.generate_node_id()
        
        self.node_id = config.node_id
        self.network_config: Optional[NetworkConfig] = None
        self.is_founder = False
        
        self.logger.info(f"Initializing node: {config.node_name} ({self.node_id[:8]}...)")
        
        self._init_components()
        
        self.logger.info("✓ Node initialization complete")
    
    def _init_components(self):
        """Initialize all node components"""
        # Peer networking
        self.peer_manager = PeerManager(self.node_id, self.config.port, self.config.host)
        
        # Consensus
        self.voting_manager = VotingManager(self.node_id)
        self.consensus_protocol = ConsensusProtocol(
            self.node_id, self.peer_manager, self.voting_manager
        )
        
        # Link voting manager to peer manager for bidirectional registration
        self.peer_manager.voting_manager = self.voting_manager
        
        # Secure aggregation
        self.aggregator = SecureAggregator(self.node_id, self.peer_manager)
        
        # Privacy
        self.budget_manager = PrivacyBudgetManager(
            self.config.epsilon_total, self.config.delta_total
        )
        self.dp_engine = DifferentialPrivacyEngine(self.budget_manager)
        
        # Audit blockchain
        self.blockchain = AuditBlockchain(self.node_id, self.config.block_difficulty)
        
        # Medical data loader (CSV or synthetic)
        if self.config.data_file:
            self.logger.info(f"Loading CSV: {self.config.data_file}")
            self.data_loader = CSVMedicalDataLoader(self.config.data_file, self.node_id)
            X, y = self.data_loader.get_full_dataset()
            self.config.num_features = X.shape[1]
            self.logger.info(f"✓ Loaded {len(X)} samples, {self.config.num_features} features")
        else:
            self.logger.info("Generating synthetic data")
            self.data_loader = MedicalDataGenerator(
                self.node_id, self.config.data_size, self.config.num_features
            )
        
        # Disease prediction model
        self.model = DiseasePredictionModel(self.config.num_features)
        
        # Anomaly detector
        self.anomaly_detector = AnomalyDetector(self.node_id)
        
        # Update local statistics
        if hasattr(self.data_loader, 'get_statistics'):
            local_stats = self.data_loader.get_statistics()
            self.anomaly_detector.set_local_statistics(local_stats)
        
        # File sharing
        self.file_sharing = SecureFileSharing(
            self.node_id, self.peer_manager, self.config.shared_files_dir
        )
        
        self.logger.debug("All components initialized")
    
    def create_network(self) -> str:
        """Create new network as founder"""
        self.is_founder = True
        network_id = CryptoHelper.generate_network_id()
        
        self.network_config = NetworkConfig(
            network_id=network_id,
            genesis_timestamp=time.time(),
            founder_node=self.node_id
        )
    
        # Register self in voting
        self.voting_manager.register_node(
            self.node_id,
            self.config.stake,
            self.config.reputation
        )
    
        # Start peer manager
        self.peer_manager.start()
        
        # Give server time to start
        time.sleep(0.5)
        
        # Log to blockchain
        self.blockchain.add_transaction({
            'type': 'network_created',
            'network_id': network_id,
            'founder': self.node_id,
            'data_source': 'csv' if self.config.data_file else 'synthetic'
        })
    
        self.logger.info(f"✓ Network created: {network_id}")
        return network_id
    
    def join_network(self, network_id: str):
        """Join existing network"""
        Validators.validate_network_id(network_id)
    
        self.network_config = NetworkConfig(
            network_id=network_id,
            genesis_timestamp=time.time(),
            founder_node='unknown'
        )
    
        # Start peer manager
        self.peer_manager.start()
    
        # Give server time to start
        time.sleep(0.5)
    
        # Register self in voting
        self.voting_manager.register_node(
            self.node_id,
            self.config.stake,
            self.config.reputation
        )
    
        # Log to blockchain
        self.blockchain.add_transaction({
            'type': 'network_joined',
            'network_id': network_id
        })
    
        self.logger.info(f"✓ Joined network: {network_id}")
    
    def connect_to_peer(self, peer_url: str, peer_id: Optional[str] = None):
        """Connect to peer with bidirectional registration - FIXED"""
        # Validate URL format
        if not peer_url.startswith('http://') and not peer_url.startswith('https://'):
            peer_url = f"http://{peer_url}"
        
        # If no peer_id provided, get it from health endpoint
        if not peer_id:
            try:
                import requests
                response = requests.get(f"{peer_url}/health", timeout=3)
                if response.status_code == 200:
                    health_data = response.json()
                    peer_id = health_data.get('node_id')
                    self.logger.debug(f"Got peer ID: {peer_id[:16]}...")
                else:
                    self.logger.error(f"Peer health check failed: HTTP {response.status_code}")
                    return False
            except Exception as e:
                self.logger.error(f"Could not reach peer at {peer_url}: {e}")
                return False
        
        # Add peer to our peer manager
        self.peer_manager.add_peer(peer_id, peer_url)
        
        # Register peer in voting with same stake
        self.voting_manager.register_node(peer_id, self.config.stake, 0.5)
        
        # NEW: Send our info to peer for bidirectional registration
        try:
            import requests
            
            # Determine our URL
            local_ip = self.peer_manager.get_local_ip()
            our_url = f"http://{local_ip}:{self.config.port}"
            
            # Send registration to peer
            registration_data = {
                'node_id': self.node_id,
                'url': our_url,
                'stake': self.config.stake,
                'reputation': self.config.reputation
            }
            
            response = requests.post(
                f"{peer_url}/register_peer",
                json=registration_data,
                timeout=3
            )
            
            if response.status_code == 200:
                self.logger.info(f"✓ Bidirectional registration successful")
            else:
                self.logger.warning(f"Peer registration response: {response.status_code}")
                
        except Exception as e:
            self.logger.warning(f"Could not complete bidirectional registration: {e}")
        
        self.logger.info(f"✓ Connected to peer: {peer_url}")
        return True
    
    def get_network_info(self) -> dict:
        """Get network information for display"""
        local_ip = self.peer_manager.get_local_ip()
        
        return {
            'node_name': self.config.node_name,
            'node_id': self.node_id,
            'local_ip': local_ip,
            'port': self.config.port,
            'connection_url': f"http://{local_ip}:{self.config.port}",
            'network_id': self.network_config.network_id if self.network_config else None,
            'is_founder': self.is_founder,
            'data_source': 'CSV' if self.config.data_file else 'Synthetic',
            'num_samples': len(self.data_loader.get_full_dataset()[0]),
            'num_features': self.config.num_features,
            'connected_peers': len(self.peer_manager.get_active_peers())
        }
    
    def run_federated_training(self, num_rounds: int = 5):
        """Run federated training rounds - FIXED with better budget checks"""
        self.logger.info(f"🚀 Starting federated training: {num_rounds} rounds")
    
        # CHECK BUDGET BEFORE STARTING
        remaining_budget = self.budget_manager.get_remaining_budget()
        if remaining_budget['epsilon_remaining'] <= 0:
            print("\n❌ ERROR: Privacy budget exhausted!")
            print("Use 'reset_budget' command before training.\n")
            return
    
        n_peers = len(self.peer_manager.get_active_peers())
        if n_peers == 0:
            self.logger.warning("⚠ No peers - running in local mode")
        else:
            self.logger.info(f"✓ Training with {n_peers} peers")
    
        # Calculate per-round budget
        epsilon_per_round = remaining_budget['epsilon_remaining'] / num_rounds
        delta_per_round = remaining_budget['delta_remaining'] / num_rounds
    
        # WARN if budget is tight
        if epsilon_per_round < 0.5:
            print(f"\n⚠ WARNING: Low privacy budget!")
            print(f"ε per round: {epsilon_per_round:.4f} (recommended: ≥1.0)")
            response = input("Continue anyway? (yes/no): ").strip().lower()
            if response != 'yes':
                print("Training cancelled.\n")
                return
    
        self.logger.info(f"Privacy budget per round: ε={epsilon_per_round:.6f}, δ={delta_per_round:.6e}")
    
        for round_num in range(1, num_rounds + 1):
            print(f"\n{'='*60}")
            print(f"ROUND {round_num}/{num_rounds}")
            print(f"{'='*60}")
            
            with OperationLogger(self.logger, f"Round {round_num}"):
                try:
                    # Step 1: Local training
                    self.logger.info("Step 1/5: Local training")
                    X, y = self.data_loader.get_full_dataset()
                    history = self.model.train_local(X, y, epochs=10)
                    
                    # Step 2: Compute statistics
                    self.logger.info("Step 2/5: Computing statistics")
                    if hasattr(self.data_loader, 'get_statistics'):
                        local_stats = self.data_loader.get_statistics()
                    else:
                        local_stats = {
                            'feature_means': np.mean(X, axis=0).tolist(),
                            'feature_stds': np.std(X, axis=0).tolist(),
                            'positive_rate': float(np.mean(y)),
                            'n_samples': len(X)
                        }
                    
                    # Step 3: Anomaly detection (if network stats available)
                    if n_peers > 0 and round_num > 1:
                        self.logger.info("Step 3/5: Anomaly detection")
                        anomaly_results = self.anomaly_detector.comprehensive_analysis(X, y)
                        self.logger.info(
                            f"Anomaly status: {anomaly_results['overall']['status']}, "
                            f"rate={anomaly_results['overall']['overall_anomaly_rate']:.2%}"
                        )
                    else:
                        anomaly_results = {'overall': {'status': 'no_network_data'}}
                    
                    # Step 4: Compute gradients with DP
                    self.logger.info("Step 4/5: Computing private gradients")
                    gradients, bias_grad = self.model.compute_gradients(X, y)
                    
                    # Clip gradients
                    clipped_grads = self.dp_engine.clip_gradients(gradients, max_norm=1.0)
                    
                    # Add DP noise
                    try:
                        private_grads = self.dp_engine.add_gaussian_noise(
                            clipped_grads,
                            sensitivity=1.0,
                            epsilon=epsilon_per_round,
                            delta=delta_per_round
                        )
                    except ValueError as e:
                        self.logger.error(f"Privacy error: {e}")
                        continue
                    
                    # Step 5: Secure aggregation
                    if n_peers > 0:
                        self.logger.info("Step 5/5: Secure aggregation")
                        if round_num == 1:
                            self.aggregator.establish_pairwise_keys()
                        
                        session_id = f"round_{round_num}_{uuid.uuid4().hex[:8]}"
                        self.aggregator.contribute(private_grads, session_id)
                        time.sleep(3)
                        aggregated_grads = self.aggregator.aggregate(timeout=10)
                        
                        if aggregated_grads is not None:
                            # Average over all nodes
                            avg_grads = aggregated_grads / (n_peers + 1)
                            self.model.update_weights(avg_grads, bias_grad, learning_rate=0.01)
                            self.logger.info("✓ Model updated with aggregated gradients")
                            
                            # Update network stats for anomaly detection
                            network_stats = local_stats.copy()
                            network_stats['feature_means'] = avg_grads.tolist()
                            self.anomaly_detector.set_network_statistics(network_stats)
                        else:
                            self.logger.warning("⚠ Aggregation failed, using local only")
                            self.model.update_weights(private_grads, bias_grad, learning_rate=0.01)
                    else:
                        self.logger.info("Step 5/5: Local update (no peers)")
                        self.model.update_weights(private_grads, bias_grad, learning_rate=0.01)
                    
                    # Consensus
                    model_params = self.model.get_parameters()
                    state_data = {
                        'round': round_num,
                        'model_weights_hash': CryptoHelper.hash_data(
                            model_params['weights'].tobytes()
                        ),
                        'anomaly_status': anomaly_results['overall']['status']
                    }
                    
                    if n_peers > 0:
                        consensus_reached = self.consensus_protocol.initiate_consensus(state_data)
                        if not consensus_reached:
                            self.logger.warning("⚠ Consensus failed for this round")
                    else:
                        consensus_reached = True
                        self.logger.info("Consensus skipped (local mode)")
                    
                    # Evaluate model
                    metrics = self.model.evaluate(X, y)
                    print(f"\n Round {round_num} Results:")
                    print(f"   Accuracy:  {metrics['accuracy']:.3f}")
                    print(f"   Precision: {metrics['precision']:.3f}")
                    print(f"   Recall:    {metrics['recall']:.3f}")
                    print(f"   F1 Score:  {metrics['f1_score']:.3f}")
                    
                    # Log to blockchain
                    self.blockchain.add_transaction({
                        'type': 'training_round',
                        'round': round_num,
                        'metrics': metrics,
                        'anomaly_summary': anomaly_results['overall'],
                        'consensus_reached': consensus_reached
                    })
                    
                except Exception as e:
                    self.logger.error(f"Error in round {round_num}: {e}", exc_info=True)
                    continue
        
        print(f"\n{'='*60}")
        print(f"✓ FEDERATED TRAINING COMPLETE")
        print(f"{'='*60}\n")
        self.logger.info("✓ Federated training complete")
    
    def reset_privacy_budget(self):
        """Reset privacy budget with confirmation"""
        confirm = input("⚠ WARNING: Type 'RESET' to confirm: ")
        if confirm == 'RESET':
            self.budget_manager.reset()
            self.logger.info("✓ Budget reset")
            print("✓ Privacy budget has been reset\n")
        else:
            self.logger.info("Cancelled")
            print("Cancelled\n")
    
    def start_interactive_shell(self):
        """Start interactive command shell"""
        self.logger.info("Interactive shell started (type 'help')")
        
        # Display connection info
        info = self.get_network_info()
        print(f"\n{'='*70}")
        print(f" AEGIS DISTRIBUTED MEDICAL NETWORK")
        print(f"{'='*70}")
        print(f"Node: {info['node_name']}")
        print(f"ID:   {info['node_id'][:16]}...")
        print(f"IP:   {info['local_ip']}:{info['port']}")
        print(f"\n Connection URL for other nodes:")
        print(f"    {info['connection_url']}")
        if info['network_id']:
            print(f"\n Network ID:")
            print(f"    {info['network_id']}")
        print(f"{'='*70}\n")
        
        # Give server time to start
        time.sleep(0.5)
        
        while True:
            try:
                command = input(f"aegis({self.config.node_name})> ").strip()
                
                if not command:
                    continue
                
                parts = command.split()
                cmd = parts[0].lower()
                args = parts[1:] if len(parts) > 1 else []
                
                if cmd in ('exit', 'quit'):
                    self.logger.info("Shutting down")
                    break
                elif cmd == 'help':
                    self._cmd_help()
                elif cmd == 'status':
                    self._cmd_status()
                elif cmd == 'network':
                    self._cmd_network_info()
                elif cmd == 'peers':
                    self._cmd_peers()
                elif cmd == 'connect':
                    self._cmd_connect()
                elif cmd == 'disconnect':
                    self._cmd_disconnect(args)
                elif cmd == 'consensus':
                    self._cmd_consensus()
                elif cmd == 'train':
                    self._cmd_train()
                elif cmd == 'audit':
                    self._cmd_audit()
                elif cmd == 'privacy':
                    self._cmd_privacy()
                elif cmd == 'model':
                    self._cmd_model()
                elif cmd == 'anomaly':
                    self._cmd_anomaly()
                elif cmd == 'share':
                    self._cmd_share_file(args)
                elif cmd == 'files':
                    self._cmd_list_files()
                elif cmd == 'download':
                    self._cmd_download_file(args)
                elif cmd == 'mine':
                    self._cmd_mine_blockchain()
                elif cmd == 'reset_budget':
                    self.reset_privacy_budget()
                elif cmd == 'reset_model':
                    self._cmd_reset_model()
                elif cmd == 'export':
                    self._cmd_export_model(args)
                elif cmd == 'clear':
                    import os
                    os.system('clear' if os.name != 'nt' else 'cls')
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.\n")
            
            except KeyboardInterrupt:
                print("\nUse 'exit' to shutdown gracefully")
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Command error: {e}", exc_info=True)
    
    def _cmd_help(self):
        """Display help information"""
        print("\n Available Commands:")
        print("="*60)
        print("  status      - Display node status")
        print("  network     - Show network connection info")
        print("  peers       - List connected peers")
        print("  connect     - Connect to a peer")
        print("  disconnect  - Disconnect from peer")
        print("  train       - Start federated training")
        print("  model       - Evaluate current model")
        print("  anomaly     - Run anomaly detection")
        print("  consensus   - Test consensus protocol")
        print("  privacy     - View privacy budget")
        print("  audit       - View blockchain audit")
        print("  mine        - Force mine pending transactions")
        print("  share <f>   - Share a file")
        print("  files       - List available files")
        print("  download <f>- Download a file")
        print("  export <p>  - Export model to file")
        print("  reset_budget- Reset privacy budget")
        print("  reset_model - Reset model weights")
        print("  clear       - Clear screen")
        print("  help        - Show this help")
        print("  exit        - Shutdown node")
        print("="*60 + "\n")
    
    def _cmd_status(self):
        """Display node status"""
        info = self.get_network_info()
        print(f"\n Node Status:")
        print("="*60)
        print(f"  Name:     {info['node_name']}")
        print(f"  ID:       {info['node_id'][:16]}...")
        print(f"  Port:     {info['port']}")
        print(f"  Network:  {info['network_id'] if info['network_id'] else 'None'}")
        print(f"  Founder:  {'Yes' if info['is_founder'] else 'No'}")
        print(f"  Data:     {info['data_source']} ({info['num_samples']} samples)")
        print(f"  Features: {info['num_features']}")
        print(f"  Peers:    {info['connected_peers']}")
        print("="*60 + "\n")
    
    def _cmd_network_info(self):
        """Display network connection information"""
        info = self.get_network_info()
        print(f"\n Network Connection Info:")
        print("="*60)
        print(f"  Local IP:  {info['local_ip']}")
        print(f"  Port:      {info['port']}")
        print(f"\n   Share this URL with other nodes:")
        print(f"     {info['connection_url']}")
        if info['network_id']:
            print(f"\n   Network ID:")
            print(f"     {info['network_id']}")
        print("="*60 + "\n")
    
    def _cmd_peers(self):
        """List connected peers"""
        active = self.peer_manager.get_active_peers()
        print(f"\n Connected Peers: {len(active)}")
        print("="*60)
        if active:
            for pid in active:
                peer = self.peer_manager.peers[pid]
                last_seen = time.time() - peer['last_seen']
                print(f"  • {pid[:16]}...")
                print(f"    URL: {peer['url']}")
                print(f"    Last seen: {last_seen:.1f}s ago")
                print(f"    Reputation: {peer['reputation']:.2f}")
        else:
            print("  No peers connected")
        print("="*60 + "\n")
    
    def _cmd_connect(self):
        """Connect to a peer"""
        print("\nConnect to Peer")
        print("="*60)
        url = input("Enter peer URL (e.g., 192.168.1.100:5001): ").strip()
        if url:
            success = self.connect_to_peer(url)
            if success:
                print("✓ Connection successful!\n")
            else:
                print("✗ Connection failed!\n")
    
    def _cmd_disconnect(self, args):
        """Disconnect from peer"""
        if not args:
            print("Usage: disconnect <peer_id_prefix>\n")
            return
        
        prefix = args[0]
        for peer_id in list(self.peer_manager.peers.keys()):
            if peer_id.startswith(prefix):
                self.peer_manager.remove_peer(peer_id)
                print(f"✓ Disconnected from {peer_id[:16]}...\n")
                return
        
        print(f"✗ No peer found with prefix: {prefix}\n")
    
    def _cmd_consensus(self):
        """Test consensus protocol"""
        print("\n  Testing consensus protocol...")
        success = self.consensus_protocol.initiate_consensus({'test': time.time()})
        if success:
            print("✓ Consensus SUCCESS\n")
        else:
            print("✗ Consensus FAILED\n")
    
    def _cmd_train(self):
        """Start federated training"""
        rounds = input("Number of rounds (default 5): ").strip()
        rounds = int(rounds) if rounds else 5
        self.run_federated_training(rounds)
    
    def _cmd_audit(self):
        """View blockchain audit"""
        print("\n  Blockchain Audit:")
        print("="*60)
        summary = self.blockchain.get_chain_summary()
        print(f"  Chain Length:    {summary['length']}")
        print(f"  Latest Hash:     {summary['latest_hash']}")
        print(f"  Transactions:    {summary['total_transactions']}")
        print(f"  Pending:         {summary['pending_transactions']}")
        
        print("\n  Recent Transactions:")
        recent = self.blockchain.get_transaction_history()[-5:]
        for tx in recent:
            print(f"    Block {tx['block_index']}: {tx['type']}")
        print("="*60 + "\n")
    
    def _cmd_mine_blockchain(self):
        """Force mine pending transactions"""
        print("\nForce mining blockchain...")
        self.blockchain.mine_pending_transactions()
        summary = self.blockchain.get_chain_summary()
        print(f"  Blockchain length: {summary['length']}")
        print(f"  Pending transactions: {summary['pending_transactions']}\n")
    
    def _cmd_privacy(self):
        """View privacy budget"""
        print("\n Privacy Budget:")
        print("="*60)
        budget = self.budget_manager.get_remaining_budget()
        print(f"  Epsilon:")
        print(f"    Total:     {self.config.epsilon_total:.4f}")
        print(f"    Remaining: {budget['epsilon_remaining']:.4f}")
        print(f"    Used:      {self.config.epsilon_total - budget['epsilon_remaining']:.4f}")
        print(f"  Delta:")
        print(f"    Total:     {self.config.delta_total:.4e}")
        print(f"    Remaining: {budget['delta_remaining']:.4e}")
        print(f"  Usage: {budget['epsilon_utilization']*100:.1f}%")
        print(f"  Queries: {budget['queries_performed']}")
        
        if budget['epsilon_utilization'] > 0.9:
            print(f"\n    ⚠ WARNING: Privacy budget nearly exhausted!")
            print(f"      Use 'reset_budget' to continue training")
        
        print("="*60 + "\n")
    
    def _cmd_model(self):
        """Evaluate model"""
        print("\n Evaluating model...")
        X, y = self.data_loader.get_full_dataset()
        metrics = self.model.evaluate(X, y)
        print("="*60)
        print(f"  Accuracy:  {metrics['accuracy']:.3f}")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1 Score:  {metrics['f1_score']:.3f}")
        print(f"  Samples:   {metrics['samples']}")
        print("="*60 + "\n")
    
    def _cmd_reset_model(self):
        """Reset model weights"""
        confirm = input("Reset model weights? (yes/no): ").strip().lower()
        if confirm == 'yes':
            self.model.reset()
            print("✓ Model weights have been reset\n")
        else:
            print("Cancelled\n")
    
    def _cmd_anomaly(self):
        """Run anomaly detection"""
        print("\n Running anomaly detection...")
        X, y = self.data_loader.get_full_dataset()
        results = self.anomaly_detector.comprehensive_analysis(X, y)
        
        print("="*60)
        print(f"  Status: {results['overall']['status'].upper()}")
        print(f"  Anomaly Rate: {results['overall']['overall_anomaly_rate']:.2%}")
        print(f"  Total: {results['overall']['total_anomalies']}/{results['n_samples']}")
        
        if 'zscore' in results['analyses']:
            print(f"\n  Z-Score Analysis:")
            print(f"    Anomalies: {results['analyses']['zscore']['n_anomalies']}")
        
        if 'label_imbalance' in results['analyses']:
            lb = results['analyses']['label_imbalance']
            if lb.get('is_anomalous'):
                print(f"\n  ⚠ WARNING: Label imbalance detected!")
                print(f"    Z-score: {lb['z_score']:.2f}")
        print("="*60 + "\n")
    
    def _cmd_share_file(self, args):
        """Share a file"""
        if not args:
            print("Usage: share <filepath>\n")
            return
        filepath = ' '.join(args)
        if Path(filepath).exists():
            success = self.file_sharing.share_file(filepath)
            print("✓ File shared!\n" if success else "✗ Failed to share file!\n")
        else:
            print(f"✗ File not found: {filepath}\n")
    
    def _cmd_list_files(self):
        """List available files"""
        files = self.file_sharing.list_available_files()
        print(f"\n Available Files: {len(files)}")
        print("="*60)
        if files:
            for f in files:
                print(f"  • {f['filename']} ({f['size_mb']:.2f} MB)")
                print(f"    Location: {f['location']}")
                print(f"    Owner: {f['owner_id'][:16]}...")
        else:
            print("  No files available")
        print("="*60 + "\n")
    
    def _cmd_download_file(self, args):
        """Download a file"""
        if not args:
            print("Usage: download <filename>\n")
            return
        filename = ' '.join(args)
        success = self.file_sharing.download_file(filename)
        print("✓ Downloading...\n" if success else "✗ Download failed!\n")
    
    def _cmd_export_model(self, args):
        """Export model to file"""
        path = ' '.join(args) if args else f"model_{self.node_id[:8]}.json"
        params = self.model.get_parameters()
        
        data = {
            'node_id': self.node_id,
            'node_name': self.config.node_name,
            'timestamp': time.time(),
            'num_features': self.config.num_features,
            'weights': params['weights'].tolist(),
            'bias': float(params['bias'])
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Exported to {path}\n")