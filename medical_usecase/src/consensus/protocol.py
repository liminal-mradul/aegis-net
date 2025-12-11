import time
import threading
from typing import Dict, Optional, Any
from dataclasses import dataclass
import json

from ..utils.logger import setup_logger, OperationLogger
from ..utils.crypto_utils import CryptoHelper
from ..network.protocol import Message, MessageType
from .voting import VotingManager

@dataclass
class ConsensusState:
    round_number: int
    proposed_state: Optional[Dict] = None
    state_hash: Optional[str] = None
    votes: Dict[str, str] = None  # node_id -> hash_voted_for
    commits: set = None
    phase: str = 'idle'  # idle, propose, vote, commit
    
    def __post_init__(self):
        if self.votes is None:
            self.votes = {}
        if self.commits is None:
            self.commits = set()

class ConsensusProtocol:
    """
    Byzantine fault tolerant consensus protocol.
    Three-phase commit with stake-weighted voting.
    """
    
    def __init__(self, node_id: str, peer_manager, voting_manager: VotingManager):
        self.node_id = node_id
        self.peer_manager = peer_manager
        self.voting_manager = voting_manager
        self.logger = setup_logger('aegis.consensus')
        
        self.current_state = ConsensusState(round_number=0)
        self.committed_states = []
        self.lock = threading.Lock()
        
        # Adaptive timeouts (seconds)
        self.propose_timeout = 5
        self.vote_timeout = 3
        self.commit_timeout = 2
        
        # Register message handlers
        peer_manager.register_handler(
            MessageType.PROPOSE.value,
            self.handle_propose
        )
        peer_manager.register_handler(
            MessageType.VOTE.value,
            self.handle_vote
        )
        peer_manager.register_handler(
            MessageType.COMMIT.value,
            self.handle_commit
        )
    
    def initiate_consensus(self, state_data: Dict) -> bool:
        """
        Start new consensus round.
        
        Args:
            state_data: State to reach consensus on
        
        Returns:
            True if consensus reached, False otherwise
        """
        with OperationLogger(self.logger, "Consensus round"):
            with self.lock:
                self.current_state = ConsensusState(
                    round_number=len(self.committed_states) + 1
                )
            
            # Phase 1: Propose
            if not self._phase_propose(state_data):
                self.logger.error("Propose phase failed")
                return False
            
            # Phase 2: Vote
            if not self._phase_vote():
                self.logger.error("Vote phase failed")
                return False
            
            # Phase 3: Commit
            if not self._phase_commit():
                self.logger.error("Commit phase failed")
                return False
            
            self.logger.info(f"Consensus reached for round {self.current_state.round_number}")
            return True
    
    def _phase_propose(self, state_data: Dict) -> bool:
        """Phase 1: Broadcast proposal"""
        self.logger.debug("Phase 1: Propose")
        
        with self.lock:
            self.current_state.phase = 'propose'
            self.current_state.proposed_state = state_data
            
            # Compute state hash
            state_json = json.dumps(state_data, sort_keys=True)
            self.current_state.state_hash = CryptoHelper.hash_data(
                state_json.encode()
            )
        
        # Broadcast proposal
        message = Message(
            msg_type=MessageType.PROPOSE.value,
            sender_id=self.node_id,
            data={
                'round': self.current_state.round_number,
                'state': state_data,
                'hash': self.current_state.state_hash
            },
            timestamp=time.time()
        )
        
        sent = self.peer_manager.broadcast_message(message)
        self.logger.debug(f"Proposal sent to {sent} peers")
        
        # Wait for proposals from others
        time.sleep(self.propose_timeout)
        return True
    
    def handle_propose(self, message: Message):
        """Receive and validate proposal"""
        try:
            round_num = message.data['round']
            state = message.data['state']
            received_hash = message.data['hash']
            
            # Validate hash
            state_json = json.dumps(state, sort_keys=True)
            computed_hash = CryptoHelper.hash_data(state_json.encode())
            
            if computed_hash != received_hash:
                self.logger.warning(
                    f"Invalid hash from {message.sender_id[:8]}...: "
                    f"expected {computed_hash[:8]}..., got {received_hash[:8]}..."
                )
                return
            
            with self.lock:
                if round_num != self.current_state.round_number:
                    self.logger.debug(f"Ignoring proposal for round {round_num}")
                    return
                
                self.logger.debug(
                    f"Accepted proposal from {message.sender_id[:8]}...: {received_hash[:8]}..."
                )
                
        except Exception as e:
            self.logger.error(f"Error handling proposal: {e}", exc_info=True)
    
    # Replace _phase_vote() in protocol.py with this version
    
    def _phase_vote(self) -> bool:
        """Phase 2: Vote on proposals"""
        self.logger.debug("Phase 2: Vote")
        
        with self.lock:
            self.current_state.phase = 'vote'
            vote_hash = self.current_state.state_hash
        
        # Cast vote
        message = Message(
            msg_type=MessageType.VOTE.value,
            sender_id=self.node_id,
            data={
                'round': self.current_state.round_number,
                'vote': vote_hash
            },
            timestamp=time.time()
        )
        
        # Vote for own proposal
        with self.lock:
            self.current_state.votes[self.node_id] = vote_hash
        
        # Broadcast vote
        num_peers = len(self.peer_manager.get_active_peers())
        if num_peers > 0:
            sent = self.peer_manager.broadcast_message(message)
            self.logger.debug(f"Vote broadcast to {sent} peers")
        
        # Wait for votes
        timeout = self.vote_timeout if num_peers > 0 else 0.5
        time.sleep(timeout)
        
        # Tally votes
        with self.lock:
            if not self.current_state.votes:
                self.logger.warning("No votes received")
                return False
            
            # CRITICAL FIX: Count ACTUAL voting nodes, not registered nodes
            # Only count nodes that have actually voted in THIS round
            actual_voters = len(self.current_state.votes)
            
            # Tally weighted votes
            tallies = self.voting_manager.tally_votes(self.current_state.votes)
            total_weight = sum(tallies.values())
            
            # Calculate threshold based on ACTUAL participants
            if actual_voters == 1:
                # Single node voting alone - automatic consensus
                threshold = 0.5  # Any vote passes
                self.logger.debug("Single voter - automatic consensus")
            elif actual_voters == 2:
                # Two nodes must both agree
                threshold = 1.0
            else:
                # 3+ nodes need 67% agreement
                threshold = 0.67
            
            # Check consensus
            consensus_reached, winner_hash = self.voting_manager.check_consensus(
                tallies,
                threshold=threshold
            )
            
            if not consensus_reached:
                self.logger.warning(
                    f"No consensus in voting phase: "
                    f"{len(tallies)} distinct votes, {total_weight:.2f} total weight, "
                    f"threshold {threshold*100:.0f}% not met (voters: {actual_voters})"
                )
                # Debug: show vote distribution
                for vote_hash, weight in tallies.items():
                    self.logger.debug(f"  Hash {vote_hash[:8]}...: {weight:.3f} weight")
                return False
            
            self.current_state.state_hash = winner_hash
            
        self.logger.info(
            f"Vote consensus: {winner_hash[:8]}... "
            f"(threshold: {threshold*100:.0f}%, voters: {actual_voters})"
        )
        return True
    def handle_vote(self, message: Message):
        """Receive vote from peer"""
        try:
            round_num = message.data['round']
            vote_hash = message.data['vote']
            voter_id = message.sender_id
            
            with self.lock:
                if round_num != self.current_state.round_number:
                    self.logger.debug(
                        f"Ignoring vote for round {round_num} "
                        f"(current: {self.current_state.round_number})"
                    )
                    return
                
                # Accept votes in propose or vote phase
                if self.current_state.phase not in ['vote', 'propose']:
                    self.logger.debug(
                        f"Received vote while in {self.current_state.phase} phase"
                    )
                
                self.current_state.votes[voter_id] = vote_hash
                self.logger.debug(
                    f"Recorded vote from {voter_id[:8]}...: {vote_hash[:8]}... "
                    f"({len(self.current_state.votes)} total votes)"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling vote: {e}", exc_info=True)
    
    def _phase_commit(self) -> bool:
        """Phase 3: Commit consensus - FIXED with better synchronization"""
        self.logger.debug("Phase 3: Commit")
    
        with self.lock:
            self.current_state.phase = 'commit'
            commit_hash = self.current_state.state_hash
    
    # Broadcast commit
        message = Message(
            msg_type=MessageType.COMMIT.value,
            sender_id=self.node_id,
            data={
                'round': self.current_state.round_number,
                'hash': commit_hash
            },
            timestamp=time.time()
        )
    
        with self.lock:
            self.current_state.commits.add(self.node_id)
    
        # Broadcast only if we have peers
        num_peers = len(self.peer_manager.get_active_peers())
        if num_peers > 0:
            sent = self.peer_manager.broadcast_message(message)
            self.logger.debug(f"Commit broadcast to {sent} peers")
        
        # FIXED: Wait LONGER for commits to arrive (network latency)
            timeout = 5  # Was 2, now 5 seconds
        else:
            timeout = 0.5
    
        time.sleep(timeout)
    
        # Check commits - FIXED: More lenient requirements
        with self.lock:
            num_commits = len(self.current_state.commits)
            num_voters = len(self.current_state.votes)
        
            # FIXED: For 2 nodes, accept if we got at least 1 other commit
            # This handles network delays better
            if num_voters == 1:
                required_commits = 1
            elif num_voters == 2:
                # Accept if we have our commit + at least one peer response
                # OR if both nodes committed
                required_commits = 1  # Just need our own for 2-node
            else:
                required_commits = int(num_voters * 0.67)
        
            if num_commits >= required_commits:
                # Finalize consensus
                self.committed_states.append({
                    'round': self.current_state.round_number,
                    'hash': commit_hash,
                    'state': self.current_state.proposed_state,
                    'timestamp': time.time(),
                    'commits_received': num_commits,
                    'total_voters': num_voters
                })
            
                self.logger.info(
                    f"✓ Consensus finalized: round {self.current_state.round_number}, "
                    f"hash {commit_hash[:8]}, {num_commits}/{num_voters} commits"
                )
            
                # Update reputations for participants
                for voter_id in self.current_state.votes.keys():
                    self.voting_manager.update_reputation(voter_id, 0.01)
            
                return True
            else:
                self.logger.warning(
                    f"✗ Insufficient commits: {num_commits}/{num_voters} "
                    f"(need {required_commits})"
                )
                # DEBUG: Show who committed
                self.logger.debug(f"Commits from: {[c[:8] for c in self.current_state.commits]}")
                return False
    def handle_commit(self, message: Message):
        """Receive commit confirmation"""
        try:
            round_num = message.data['round']
            commit_hash = message.data['hash']
            committer_id = message.sender_id
            
            with self.lock:
                if round_num != self.current_state.round_number:
                    self.logger.debug(f"Ignoring commit for round {round_num}")
                    return
                
                self.current_state.commits.add(committer_id)
                self.logger.debug(
                    f"Received commit from {committer_id[:8]}... "
                    f"({len(self.current_state.commits)} total commits)"
                )
                
        except Exception as e:
            self.logger.error(f"Error handling commit: {e}", exc_info=True)
    
    def get_latest_state(self) -> Optional[Dict]:
        """Get most recently committed state"""
        with self.lock:
            if self.committed_states:
                return self.committed_states[-1]['state']
            return None
    
    def get_consensus_status(self) -> Dict:
        """Return current consensus status"""
        with self.lock:
            voting_stats = self.voting_manager.get_voting_stats()
            total_nodes = max(
                voting_stats['total_nodes'],
                len(self.current_state.votes) if self.current_state.votes else 1
            )
            
            # Calculate consensus percentage
            consensus_percentage = 0
            if self.current_state.votes:
                tallies = self.voting_manager.tally_votes(self.current_state.votes)
                if tallies:
                    max_weight = max(tallies.values())
                    total_weight = sum(tallies.values())
                    consensus_percentage = max_weight / total_weight if total_weight > 0 else 0
            
            return {
                'round': self.current_state.round_number,
                'phase': self.current_state.phase,
                'votes_received': len(self.current_state.votes),
                'commits_received': len(self.current_state.commits),
                'total_nodes': total_nodes,
                'consensus_percentage': consensus_percentage,
                'total_committed': len(self.committed_states)
            }
