import numpy as np
from typing import Union
from ..utils.logger import setup_logger
from .budget import PrivacyBudgetManager

class DifferentialPrivacyEngine:
    def __init__(self, budget_manager: PrivacyBudgetManager):
        self.budget_manager = budget_manager
        self.logger = setup_logger('aegis.privacy.dp')
    
    def add_laplace_noise(self, value: Union[float, np.ndarray], 
                         sensitivity: float, epsilon: float) -> Union[float, np.ndarray]:
        """Add Laplace noise for pure ε-DP"""
        # Check budget
        available, msg = self.budget_manager.check_budget(epsilon, 0)
        if not available:
            raise ValueError(f"Privacy budget exhausted: {msg}")
        
        # Laplace mechanism: scale = Δf / ε
        scale = sensitivity / epsilon
        
        if isinstance(value, np.ndarray):
            noise = np.random.laplace(0, scale, size=value.shape)
            noisy_value = value + noise
        else:
            noise = np.random.laplace(0, scale)
            noisy_value = value + noise
        
        self.budget_manager.consume_budget(epsilon, 0, 'Laplace')
        
        self.logger.debug(f"Laplace: Δf={sensitivity}, ε={epsilon}, scale={scale:.4f}")
        return noisy_value
    
    def add_gaussian_noise(self, value: Union[float, np.ndarray],
                          sensitivity: float, epsilon: float, delta: float) -> Union[float, np.ndarray]:
        """Add Gaussian noise for (ε,δ)-DP - FIXED"""
        # FIXED: Better budget checking and error messages
        if epsilon <= 0:
            raise ValueError(
                "Privacy budget exhausted: epsilon must be > 0. "
                "Reset budget with 'reset_budget' command."
            )
        
        if delta <= 0:
            raise ValueError(
                "Privacy budget exhausted: delta must be > 0. "
                "Reset budget with 'reset_budget' command."
            )
        
        available, msg = self.budget_manager.check_budget(epsilon, delta)
        if not available:
            # Try to use remaining budget
            remaining_eps = self.budget_manager.get_remaining_epsilon()
            remaining_delta = self.budget_manager.get_remaining_delta()
            
            if remaining_eps > 0 and remaining_delta > 0:
                self.logger.warning(
                    f"Requested ε={epsilon:.4f}, δ={delta:.4e} exceeds budget. "
                    f"Using remaining: ε={remaining_eps:.4f}, δ={remaining_delta:.4e}"
                )
                epsilon = remaining_eps
                delta = remaining_delta
            else:
                raise ValueError(
                    f"Privacy budget exhausted: {msg}. "
                    "Reset budget with 'reset_budget' command."
                )
        
        # Gaussian mechanism: σ = (Δf * sqrt(2 * ln(1.25/δ))) / ε
        sigma = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        if isinstance(value, np.ndarray):
            noise = np.random.normal(0, sigma, size=value.shape)
            noisy_value = value + noise
        else:
            noise = np.random.normal(0, sigma)
            noisy_value = value + noise
        
        self.budget_manager.consume_budget(epsilon, delta, 'Gaussian')
        
        self.logger.debug(f"Gaussian: Δf={sensitivity}, ε={epsilon:.4f}, δ={delta:.4e}, σ={sigma:.4f}")
        return noisy_value
    
    def clip_gradients(self, gradients: np.ndarray, max_norm: float) -> np.ndarray:
        """L2 norm clipping for gradients"""
        grad_norm = np.linalg.norm(gradients)
        
        if grad_norm > max_norm:
            clipped = gradients * (max_norm / grad_norm)
            self.logger.debug(f"Clipped: ||g||={grad_norm:.4f} → {max_norm:.4f}")
            return clipped
        
        return gradients
    
    def get_budget_status(self) -> dict:
        """Get current budget status"""
        return self.budget_manager.get_remaining_budget()