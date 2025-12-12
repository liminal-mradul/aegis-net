import threading
from typing import Dict, Tuple
import numpy as np

from ..utils.logger import setup_logger
from ..utils.validators import Validators, ValidationError

class VotingManager:
    """
    Thread-safe voting manager with stake-weighted voting.
    Implements reputation-based consensus mechanism.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.stakes: Dict[str, float] = {}
        self.reputations: Dict[str, float] = {}
        self.logger = setup_logger('aegis.voting')
        
        # Thread safety lock
        self._lock = threading.Lock()
    
    def register_node(self, node_id: str, stake: float, reputation: float = 0.5):
        """Register node with stake and reputation (thread-safe)"""
        try:
            Validators.validate_stake(stake)
            Validators.validate_reputation(reputation)
        except ValidationError as e:
            self.logger.error(f"Invalid registration parameters: {e}")
            raise
        
        with self._lock:
            self.stakes[node_id] = stake
            self.reputations[node_id] = reputation
            self.logger.debug(f"Registered node {node_id[:16]}...: stake={stake}, rep={reputation:.3f}")
    
    def unregister_node(self, node_id: str):
        """Remove node from voting (thread-safe)"""
        with self._lock:
            if node_id in self.stakes:
                del self.stakes[node_id]
            if node_id in self.reputations:
                del self.reputations[node_id]
            self.logger.debug(f"Unregistered node {node_id[:16]}...")
    
    def get_voting_power(self, node_id: str) -> float:
        """Calculate normalized voting power (thread-safe)"""
        with self._lock:
            return self._get_voting_power_unsafe(node_id)
    
    def _get_voting_power_unsafe(self, node_id: str) -> float:
        """Internal: Get voting power without acquiring lock"""
        if node_id not in self.stakes:
            return 0.0
        
        stake = self.stakes[node_id]
        reputation = self.reputations.get(node_id, 0.5)
        
        # Voting power = stake * reputation
        weighted_power = stake * reputation
        
        # Normalize across all nodes
        total_weighted = sum(
            self.stakes[nid] * self.reputations.get(nid, 0.5)
            for nid in self.stakes
        )
        
        if total_weighted == 0:
            return 0.0
        
        return weighted_power / total_weighted
    
    def tally_votes(self, votes: Dict[str, str]) -> Dict[str, float]:
        """Tally weighted votes for each candidate (thread-safe)"""
        tallies = {}
        
        with self._lock:
            for node_id, candidate in votes.items():
                voting_power = self._get_voting_power_unsafe(node_id)
                tallies[candidate] = tallies.get(candidate, 0.0) + voting_power
        
        self.logger.debug(f"Vote tallies: {len(tallies)} candidates")
        return tallies
    
    def check_consensus(self, tallies: Dict[str, float], 
                       threshold: float = 0.67) -> Tuple[bool, str]:
        """Check if any candidate exceeds consensus threshold"""
        if not tallies:
            return False, None
        
        # Find candidate with maximum votes
        max_candidate = max(tallies.items(), key=lambda x: x[1])
        candidate_hash, weight = max_candidate
        
        if weight >= threshold:
            self.logger.info(
                f"Consensus reached: {candidate_hash[:16]}... with {weight:.2%} "
                f"(threshold: {threshold:.2%})"
            )
            return True, candidate_hash
        
        self.logger.debug(
            f"No consensus: max vote {weight:.2%} < threshold {threshold:.2%}"
        )
        return False, None
    
    def update_reputation(self, node_id: str, delta: float):
        """Update node reputation (thread-safe)"""
        with self._lock:
            if node_id in self.reputations:
                old_rep = self.reputations[node_id]
                self.reputations[node_id] = np.clip(old_rep + delta, 0.0, 1.0)
                self.logger.debug(
                    f"Updated reputation for {node_id[:16]}...: "
                    f"{old_rep:.3f} -> {self.reputations[node_id]:.3f}"
                )
    
    def get_reputation(self, node_id: str) -> float:
        """Get node reputation (thread-safe)"""
        with self._lock:
            return self.reputations.get(node_id, 0.5)
    
    def get_stake(self, node_id: str) -> float:
        """Get node stake (thread-safe)"""
        with self._lock:
            return self.stakes.get(node_id, 0.0)
    
    def get_byzantine_tolerance(self) -> int:
        """Calculate Byzantine fault tolerance threshold (internal use)"""
        # CRITICAL: This should NOT acquire lock - called from locked context
        n = len(self.stakes)
        return int(n / 3)
    
    def get_voting_stats(self) -> Dict:
        """Get voting statistics (thread-safe)"""
        with self._lock:
            # Call _get_byzantine_tolerance_unsafe to avoid deadlock
            byzantine_tolerance = int(len(self.stakes) / 3)
            
            return {
                'total_nodes': len(self.stakes),
                'total_stake': sum(self.stakes.values()),
                'avg_reputation': np.mean(list(self.reputations.values())) if self.reputations else 0.5,
                'byzantine_tolerance': byzantine_tolerance
            }
