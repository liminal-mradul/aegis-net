import time
import json
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

from ..utils.logger import setup_logger
from ..utils.crypto_utils import CryptoHelper
from .merkle import MerkleTree

@dataclass
class Block:
    index: int
    timestamp: float
    transactions: List[Dict]
    merkle_root: str
    previous_hash: str
    nonce: int
    hash: str

class AuditBlockchain:
    """
    Thread-safe audit blockchain with proof-of-work mining.
    Maintains immutable transaction history.
    """
    
    def __init__(self, node_id: str, difficulty: int = 4):
        self.node_id = node_id
        self.difficulty = difficulty
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict] = []
        self.logger = setup_logger('aegis.audit')
        
        # Thread safety locks
        self._chain_lock = threading.Lock()
        self._pending_lock = threading.Lock()
        self._mining_lock = threading.Lock()
        
        # Mining state
        self._is_mining = False
        
        # Create genesis block
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create first block in chain"""
        genesis = Block(
            index=0,
            timestamp=time.time(),
            transactions=[{'type': 'genesis', 'node_id': self.node_id}],
            merkle_root='0' * 64,
            previous_hash='0' * 64,
            nonce=0,
            hash='0' * 64
        )
        
        # Mine genesis block
        genesis.hash = self._mine_block(genesis)
        
        with self._chain_lock:
            self.chain.append(genesis)
        
        self.logger.info("Genesis block created")
    
    def _mine_block(self, block: Block) -> str:
        """
        Proof-of-work mining.
        Find nonce such that hash starts with 'difficulty' zeros.
        """
        target = '0' * self.difficulty
        attempts = 0
        
        while True:
            block_data = (
                f"{block.index}{block.timestamp}{block.merkle_root}"
                f"{block.previous_hash}{block.nonce}"
            )
            block_hash = CryptoHelper.hash_data(block_data.encode())
            
            if block_hash.startswith(target):
                self.logger.debug(
                    f"Block mined: {block_hash[:16]}... (nonce={block.nonce}, "
                    f"attempts={attempts})"
                )
                return block_hash
            
            block.nonce += 1
            attempts += 1
            
            # Progress indicator every 500k attempts (less spam)
            if attempts > 0 and attempts % 500000 == 0:
                self.logger.debug(f"Mining progress: {attempts:,} attempts")
    
    def add_transaction(self, transaction: Dict):
        """
        Add transaction to pending pool (thread-safe).
        Automatically mines block when enough transactions accumulated.
        """
        transaction['timestamp'] = time.time()
        
        with self._pending_lock:
            self.pending_transactions.append(transaction)
            num_pending = len(self.pending_transactions)
        
        # Mine block if enough transactions (10 transactions per block)
        if num_pending >= 10:
            self.mine_pending_transactions()
    
    def mine_pending_transactions(self):
        """
        Mine block with pending transactions (thread-safe).
        Prevents concurrent mining with lock.
        """
        # Prevent concurrent mining
        if not self._mining_lock.acquire(blocking=False):
            self.logger.debug("Mining already in progress, skipping")
            return
        
        try:
            with self._pending_lock:
                if not self.pending_transactions:
                    return
                
                # Copy pending transactions
                transactions_to_mine = self.pending_transactions.copy()
                num_tx = len(transactions_to_mine)
            
            self.logger.info(f"Mining block with {num_tx} transactions")
            
            # Build Merkle tree
            tx_strings = [json.dumps(tx, sort_keys=True) for tx in transactions_to_mine]
            merkle_tree = MerkleTree(tx_strings)
            
            # Get previous hash safely
            with self._chain_lock:
                previous_hash = self.chain[-1].hash
                new_index = len(self.chain)
            
            # Create new block
            new_block = Block(
                index=new_index,
                timestamp=time.time(),
                transactions=transactions_to_mine,
                merkle_root=merkle_tree.root,
                previous_hash=previous_hash,
                nonce=0,
                hash=''
            )
            
            # Mine block (this can take time)
            new_block.hash = self._mine_block(new_block)
            
            # Add to chain and clear pending
            with self._chain_lock:
                self.chain.append(new_block)
            
            with self._pending_lock:
                # Only clear transactions that were mined
                self.pending_transactions = self.pending_transactions[num_tx:]
            
            self.logger.info(
                f"Block {new_block.index} added to chain: {new_block.hash[:16]}..."
            )
            
        finally:
            self._mining_lock.release()
    # Add this method to blockchain.py
    
    def add_transaction(self, transaction: Dict, auto_mine: bool = True):
        """
        Add transaction to pending pool (thread-safe).
        
        Args:
            transaction: Transaction data
            auto_mine: If True, mines block when threshold reached.
                      If False, only queues transaction.
        """
        transaction['timestamp'] = time.time()
        
        with self._pending_lock:
            self.pending_transactions.append(transaction)
            num_pending = len(self.pending_transactions)
        
        # Auto-mine if enabled and threshold reached
        # OR if this is important transaction and we have some pending
        important_types = {'training_round', 'consensus_finalized', 'network_joined'}
        
        if auto_mine:
            should_mine = (
                num_pending >= 10 or  # Normal threshold
                (transaction.get('type') in important_types and num_pending >= 3)  # Important tx
            )
            
            if should_mine:
                self.mine_pending_transactions()
    
    def force_mine(self):
        """
        Force mining of pending transactions regardless of count.
        Useful for ensuring transactions are recorded.
        """
        with self._pending_lock:
            if not self.pending_transactions:
                self.logger.info("No pending transactions to mine")
                return
        
        self.logger.info("Force mining pending transactions")
        self.mine_pending_transactions()
    
    def verify_chain(self) -> bool:
        """
        Verify entire blockchain integrity (thread-safe).
        Checks hash linkage and proof-of-work for all blocks.
        """
        with self._chain_lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i-1]
                
                # Check previous hash linkage
                if current.previous_hash != previous.hash:
                    self.logger.error(f"Chain broken at block {i}: hash mismatch")
                    return False
                
                # Recompute hash
                block_data = (
                    f"{current.index}{current.timestamp}{current.merkle_root}"
                    f"{current.previous_hash}{current.nonce}"
                )
                computed_hash = CryptoHelper.hash_data(block_data.encode())
                
                if computed_hash != current.hash:
                    self.logger.error(f"Invalid hash at block {i}")
                    return False
                
                # Verify proof-of-work
                target = '0' * self.difficulty
                if not current.hash.startswith(target):
                    self.logger.error(f"Invalid proof-of-work at block {i}")
                    return False
        
        self.logger.info("Blockchain verification passed")
        return True
    
    def get_transaction_history(self, tx_type: Optional[str] = None) -> List[Dict]:
        """
        Query transaction history (thread-safe).
        
        Args:
            tx_type: Filter by transaction type (optional)
        
        Returns:
            List of transactions with block metadata
        """
        all_txs = []
        
        with self._chain_lock:
            for block in self.chain:
                for tx in block.transactions:
                    if tx_type is None or tx.get('type') == tx_type:
                        all_txs.append({
                            'block_index': block.index,
                            'block_hash': block.hash[:16],
                            'block_timestamp': block.timestamp,
                            **tx
                        })
        
        return all_txs
    
    def get_block(self, index: int) -> Optional[Block]:
        """Get block by index (thread-safe)"""
        with self._chain_lock:
            if 0 <= index < len(self.chain):
                return self.chain[index]
            return None
    
    def get_latest_block(self) -> Block:
        """Get latest block (thread-safe)"""
        with self._chain_lock:
            return self.chain[-1]
    
    def get_chain_summary(self) -> Dict:
        """Get blockchain summary statistics (thread-safe)"""
        with self._chain_lock, self._pending_lock:
            return {
                'length': len(self.chain),
                'latest_hash': self.chain[-1].hash[:16] if self.chain else None,
                'latest_index': self.chain[-1].index if self.chain else None,
                'pending_transactions': len(self.pending_transactions),
                'total_transactions': sum(len(b.transactions) for b in self.chain),
                'difficulty': self.difficulty
            }
