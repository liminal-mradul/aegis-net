import time
import threading
from typing import Dict, List, Tuple
from ..utils.logger import setup_logger

class PrivacyBudgetManager:
    """
    Thread-safe privacy budget manager for differential privacy.
    Tracks epsilon and delta consumption across queries.
    """
    
    def __init__(self, epsilon_total: float, delta_total: float):
        self.epsilon_total = epsilon_total
        self.delta_total = delta_total
        self.epsilon_used = 0.0
        self.delta_used = 0.0
        self.audit_log: List[Dict] = []
        self.logger = setup_logger('aegis.privacy.budget')
        
        # Thread safety lock
        self._lock = threading.Lock()
        
        self.logger.info(f"Budget initialized: ε={epsilon_total}, δ={delta_total}")
    
    def check_and_consume_budget(self, epsilon_request: float, 
                                  delta_request: float, 
                                  mechanism: str) -> Tuple[bool, str]:
        """
        Atomically check and consume privacy budget.
        This prevents race conditions between check and consume.
        
        Args:
            epsilon_request: Epsilon to consume
            delta_request: Delta to consume
            mechanism: Name of DP mechanism
        
        Returns:
            (success, message) tuple
        """
        with self._lock:
            # Check budget availability
            if self.epsilon_used + epsilon_request > self.epsilon_total:
                return False, (
                    f"Epsilon exhausted: {self.epsilon_used:.4f}/{self.epsilon_total:.4f}, "
                    f"requested {epsilon_request:.4f}"
                )
            
            if self.delta_used + delta_request > self.delta_total:
                return False, (
                    f"Delta exhausted: {self.delta_used:.4e}/{self.delta_total:.4e}, "
                    f"requested {delta_request:.4e}"
                )
            
            # Consume budget
            self.epsilon_used += epsilon_request
            self.delta_used += delta_request
            
            # Log consumption
            entry = {
                'timestamp': time.time(),
                'mechanism': mechanism,
                'epsilon': epsilon_request,
                'delta': delta_request,
                'total_epsilon': self.epsilon_used,
                'total_delta': self.delta_used
            }
            self.audit_log.append(entry)
            
            self.logger.debug(
                f"{mechanism}: ε={epsilon_request:.4f}, δ={delta_request:.4e}, "
                f"remaining: ε={self.epsilon_total - self.epsilon_used:.4f}"
            )
            
            return True, "Budget consumed"
    
    def check_budget(self, epsilon_request: float, delta_request: float) -> Tuple[bool, str]:
        """
        Check if budget is available (non-consuming check).
        Note: Use check_and_consume_budget for atomic operations.
        """
        with self._lock:
            if self.epsilon_used + epsilon_request > self.epsilon_total:
                return False, f"Epsilon exhausted: {self.epsilon_used:.4f}/{self.epsilon_total:.4f}"
            
            if self.delta_used + delta_request > self.delta_total:
                return False, f"Delta exhausted: {self.delta_used:.4e}/{self.delta_total:.4e}"
            
            return True, "Budget available"
    
    def consume_budget(self, epsilon: float, delta: float, mechanism: str):
        """
        Consume budget (for backward compatibility).
        WARNING: Not atomic with check_budget. Use check_and_consume_budget instead.
        """
        with self._lock:
            self.epsilon_used += epsilon
            self.delta_used += delta
            
            entry = {
                'timestamp': time.time(),
                'mechanism': mechanism,
                'epsilon': epsilon,
                'delta': delta,
                'total_epsilon': self.epsilon_used,
                'total_delta': self.delta_used
            }
            self.audit_log.append(entry)
            
            self.logger.debug(
                f"{mechanism}: ε={epsilon:.4f}, δ={delta:.4e}, "
                f"remaining: {self.epsilon_total - self.epsilon_used:.4f}"
            )
    
    def get_remaining_budget(self) -> Dict:
        """Get remaining privacy budget (thread-safe)"""
        with self._lock:
            epsilon_remaining = max(0.0, self.epsilon_total - self.epsilon_used)
            delta_remaining = max(0.0, self.delta_total - self.delta_used)
            
            return {
                'epsilon_remaining': epsilon_remaining,
                'delta_remaining': delta_remaining,
                'epsilon_utilization': self.epsilon_used / self.epsilon_total if self.epsilon_total > 0 else 0,
                'queries_performed': len(self.audit_log)
            }
    
    def get_remaining_epsilon(self) -> float:
        """Get remaining epsilon (thread-safe)"""
        with self._lock:
            return max(0.0, self.epsilon_total - self.epsilon_used)
    
    def get_remaining_delta(self) -> float:
        """Get remaining delta (thread-safe)"""
        with self._lock:
            return max(0.0, self.delta_total - self.delta_used)
    
    def reset(self):
        """Reset privacy budget (use with caution)"""
        with self._lock:
            self.logger.warning("Privacy budget reset - all consumption history cleared")
            self.epsilon_used = 0.0
            self.delta_used = 0.0
            self.audit_log.clear()
    
    def get_audit_log(self) -> List[Dict]:
        """Get copy of audit log (thread-safe)"""
        with self._lock:
            return self.audit_log.copy()
