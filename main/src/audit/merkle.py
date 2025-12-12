import hashlib
from typing import List, Tuple, Optional

class MerkleTree:
    """
    Merkle tree for efficient transaction verification.
    Provides O(log n) proof size for transaction inclusion.
    """
    
    def __init__(self, transactions: List[str]):
        if not transactions:
            raise ValueError("Cannot create Merkle tree with empty transaction list")
        
        self.transactions = transactions
        self.tree = self._build_tree()
        self.root = self.tree[0][0] if self.tree else None
    
    def _hash(self, data: str) -> str:
        """Compute SHA3-256 hash"""
        return hashlib.sha3_256(data.encode()).hexdigest()
    
    def _build_tree(self) -> List[List[str]]:
        """
        Build Merkle tree bottom-up.
        Returns list of levels, with root at index 0.
        """
        if not self.transactions:
            return []
        
        # Leaf level - hash each transaction
        current_level = [self._hash(tx) for tx in self.transactions]
        tree = [current_level]
        
        # Build tree bottom-up until single root
        while len(current_level) > 1:
            next_level = []
            
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                
                # Handle odd number of nodes
                if i + 1 < len(current_level):
                    right = current_level[i + 1]
                else:
                    # Duplicate last node for odd-length level
                    right = left
                
                # Hash concatenation of children
                parent = self._hash(left + right)
                next_level.append(parent)
            
            tree.insert(0, next_level)
            current_level = next_level
        
        return tree
    
    def get_proof(self, tx_index: int) -> List[Tuple[str, str]]:
        """
        Generate Merkle proof for transaction at given index.
        
        Args:
            tx_index: Index of transaction in original list
        
        Returns:
            List of (sibling_hash, position) tuples
            Position is 'left' or 'right' indicating sibling position
        
        Raises:
            IndexError: If transaction index is out of range
        """
        if tx_index < 0 or tx_index >= len(self.transactions):
            raise IndexError(
                f"Transaction index {tx_index} out of range [0, {len(self.transactions)})"
            )
        
        proof = []
        index = tx_index
        
        # Traverse from leaf to root, collecting sibling hashes
        for level in reversed(self.tree[1:]):  # Skip root level
            # Determine sibling index
            if index % 2 == 0:
                # Current node is left child
                sibling_index = index + 1
                position = 'right'
            else:
                # Current node is right child
                sibling_index = index - 1
                position = 'left'
            
            # Add sibling to proof if it exists
            if sibling_index < len(level):
                proof.append((level[sibling_index], position))
            
            # Move to parent index
            index = index // 2
        
        return proof
    
    def verify_proof(self, tx: str, proof: List[Tuple[str, str]], root: str) -> bool:
        """
        Verify Merkle proof for transaction.
        
        Args:
            tx: Original transaction string
            proof: Merkle proof from get_proof()
            root: Expected Merkle root
        
        Returns:
            True if proof is valid, False otherwise
        """
        # Start with hash of transaction
        current_hash = self._hash(tx)
        
        # Apply each proof step
        for sibling_hash, position in proof:
            if position == 'right':
                # Sibling is on right, current is on left
                current_hash = self._hash(current_hash + sibling_hash)
            else:
                # Sibling is on left, current is on right
                current_hash = self._hash(sibling_hash + current_hash)
        
        # Check if computed root matches expected root
        return current_hash == root
    
    def get_tree_height(self) -> int:
        """Get height of Merkle tree"""
        return len(self.tree)
    
    def get_leaf_count(self) -> int:
        """Get number of leaf nodes (transactions)"""
        return len(self.transactions)
