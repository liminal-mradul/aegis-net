import numpy as np
from typing import Tuple
from sklearn.datasets import make_classification

from ..utils.logger import setup_logger

class MedicalDataGenerator:
    def __init__(self, node_id: str, data_size: int = 1000, num_features: int = 20):
        self.node_id = node_id
        self.data_size = data_size
        self.num_features = num_features
        self.logger = setup_logger('aegis.medical.data')
        
        # Generate synthetic patient data
        self.X, self.y = self._generate_data()
        
        self.logger.info(
            f"Generated synthetic medical dataset: "
            f"{self.data_size} samples, {self.num_features} features"
        )
    
    def _generate_data(self) -> Tuple[np.ndarray, np.ndarray]:
        # Generate synthetic classification dataset
        # Simulates disease prediction from patient vitals/biomarkers
        
        # Use node_id hash as random seed for reproducibility
        seed = int(self.node_id[:8], 16) % (2**32)
        
        X, y = make_classification(
            n_samples=self.data_size,
            n_features=self.num_features,
            n_informative=15,
            n_redundant=3,
            n_classes=2,
            weights=[0.7, 0.3],  # Imbalanced (30% disease prevalence)
            flip_y=0.05,  # 5% label noise
            random_state=seed
        )
        
        # Scale features to realistic medical ranges
        X = self._scale_to_medical_ranges(X)
        
        return X, y
    
    def _scale_to_medical_ranges(self, X: np.ndarray) -> np.ndarray:
        # Scale features to realistic medical value ranges
        # Example features: blood pressure, heart rate, glucose, etc.
        
        feature_ranges = [
            (90, 180),    # Systolic BP
            (60, 120),    # Diastolic BP
            (60, 120),    # Heart rate
            (70, 200),    # Glucose
            (35, 42),     # Body temp (C)
            (16, 25),     # Respiratory rate
            (90, 100),    # O2 saturation
        ] + [(0, 100)] * (self.num_features - 7)  # Other normalized biomarkers
        
        scaled = np.zeros_like(X)
        for i in range(min(len(feature_ranges), self.num_features)):
            min_val, max_val = feature_ranges[i]
            # Normalize to [0,1] then scale to range
            col = X[:, i]
            col_norm = (col - col.min()) / (col.max() - col.min() + 1e-8)
            scaled[:, i] = col_norm * (max_val - min_val) + min_val
        
        return scaled
    
    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        # Get random batch for training
        indices = np.random.choice(self.data_size, batch_size, replace=False)
        return self.X[indices], self.y[indices]
    
    def get_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y
    
    def get_statistics(self) -> dict:
        # Return dataset statistics without revealing individual records
        return {
            'size': self.data_size,
            'features': self.num_features,
            'positive_class_ratio': float(np.mean(self.y)),
            'feature_means': np.mean(self.X, axis=0).tolist(),
            'feature_stds': np.std(self.X, axis=0).tolist()
        }
