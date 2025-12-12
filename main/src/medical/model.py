import numpy as np
from typing import Tuple
from sklearn.metrics import roc_auc_score

from ..utils.logger import setup_logger

class DiseasePredictionModel:
    def __init__(self, num_features: int):
        self.num_features = num_features
        # FIXED: Xavier/Glorot initialization for stability
        self.weights = np.random.randn(num_features) * np.sqrt(2.0 / num_features)
        self.bias = 0.0
        self.logger = setup_logger('aegis.medical.model')
        
        self.logger.info(f"Initialized logistic regression model: {num_features} features")
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid - FIXED TO PREVENT OVERFLOW"""
        # Clip extreme values to prevent overflow
        z = np.clip(z, -500, 500)
        
        # Use numerically stable formula
        return np.where(
            z >= 0,
            1 / (1 + np.exp(-z)),
            np.exp(z) / (1 + np.exp(z))
        )
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of positive class"""
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Binary prediction"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute gradients for logistic regression"""
        m = len(y)
        
        # Forward pass
        predictions = self.predict_proba(X)
        
        # Compute gradients
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        
        return dw, db
    
    def update_weights(self, gradients: np.ndarray, bias_gradient: float, learning_rate: float = 0.01):
        """Update model parameters"""
        self.weights -= learning_rate * gradients
        self.bias -= learning_rate * bias_gradient
    
    def train_local(self, X: np.ndarray, y: np.ndarray, 
                   epochs: int = 10, learning_rate: float = 0.05) -> dict:
        """Local training - FIXED with gradient clipping and early stopping"""
        self.logger.debug(f"Starting local training: {epochs} epochs")
        
        history = {'loss': [], 'accuracy': []}
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(epochs):
            # Compute loss
            predictions = self.predict_proba(X)
            loss = -np.mean(y * np.log(predictions + 1e-8) + 
                          (1 - y) * np.log(1 - predictions + 1e-8))
            history['loss'].append(loss)
            
            # Compute accuracy
            acc = np.mean((predictions >= 0.5).astype(int) == y)
            history['accuracy'].append(acc)
            
            # Early stopping
            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    self.logger.debug(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # FIXED: Clip gradients to prevent explosion
            dw = np.clip(dw, -1.0, 1.0)
            db = np.clip(db, -1.0, 1.0)
            
            # Update weights
            self.update_weights(dw, db, learning_rate)
            
            if (epoch + 1) % 10 == 0:
                self.logger.debug(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Acc: {acc:.4f}")
        
        return history
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance"""
        predictions = self.predict(X)
        proba = self.predict_proba(X)
        
        accuracy = np.mean(predictions == y)
        
        tp = np.sum((predictions == 1) & (y == 1))
        fp = np.sum((predictions == 1) & (y == 0))
        tn = np.sum((predictions == 0) & (y == 0))
        fn = np.sum((predictions == 0) & (y == 1))
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        try:
            auc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else 0.0
        except:
            auc = 0.0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity),
            'auc_roc': float(auc),
            'samples': len(y),
            'tp': int(tp),
            'fp': int(fp),
            'tn': int(tn),
            'fn': int(fn)
        }
    
    def get_parameters(self) -> dict:
        return {
            'weights': self.weights.copy(),
            'bias': self.bias
        }
    
    def set_parameters(self, params: dict):
        self.weights = params['weights'].copy()
        self.bias = params['bias']
    
    def reset(self):
        """Reset model to Xavier initialization"""
        self.weights = np.random.randn(self.num_features) * np.sqrt(2.0 / self.num_features)
        self.bias = 0.0
        self.logger.info("Model weights reset to Xavier initialization")

