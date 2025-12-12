import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
from ..utils.logger import setup_logger

class CSVMedicalDataLoader:
    REQUIRED_COLUMNS = ['patient_id', 'diagnosis']
    
    def __init__(self, csv_path: str, node_id: str):
        self.csv_path = Path(csv_path)
        self.node_id = node_id
        self.logger = setup_logger('aegis.csv_loader')
        
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.metadata = {}
        
        self._load_and_validate()
    
    def _load_and_validate(self):
        try:
            self.logger.info(f"Loading CSV: {self.csv_path}")
            self.df = pd.read_csv(self.csv_path)
            self.logger.debug(f"Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
            
            # Check required columns
            missing = set(self.REQUIRED_COLUMNS) - set(self.df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            
            self._process_diagnosis()
            self._extract_features()
            self._extract_metadata()
            
            self.logger.info(f"Data loaded: {len(self.X)} samples, {len(self.feature_names)} features")
            
        except Exception as e:
            self.logger.error(f"Failed to load CSV: {e}", exc_info=True)
            raise
    
    def _process_diagnosis(self):
        diag_col = self.df['diagnosis']
        
        if pd.api.types.is_numeric_dtype(diag_col):
            self.y = diag_col.values.astype(int)
            # Ensure binary
            if not set(np.unique(self.y)).issubset({0, 1}):
                self.logger.warning("Converting non-binary diagnosis to binary")
                self.y = (self.y > 0).astype(int)
        else:
            positive_labels = {'positive', 'yes', 'diseased', 'abnormal', '1', 'true'}
            self.y = diag_col.str.lower().str.strip().isin(positive_labels).astype(int)
            self.logger.debug(f"Labels: {np.sum(self.y)} positive, {len(self.y) - np.sum(self.y)} negative")
    
    def _extract_features(self):
        exclude_cols = set(self.REQUIRED_COLUMNS + ['age', 'gender', 'admission_date'])
        
        feature_cols = []
        for col in self.df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(self.df[col]):
                feature_cols.append(col)
        
        if not feature_cols:
            raise ValueError("No numerical feature columns found")
        
        self.feature_names = feature_cols
        self.X = self.df[feature_cols].values
        
        # Handle missing values
        if np.any(np.isnan(self.X)):
            n_missing = np.sum(np.isnan(self.X))
            self.logger.warning(f"Found {n_missing} missing values, filling with column means")
            col_means = np.nanmean(self.X, axis=0)
            for i in range(self.X.shape[1]):
                mask = np.isnan(self.X[:, i])
                self.X[mask, i] = col_means[i]
        
        # Standardize: z = (x - μ) / σ
        # MATH VERIFIED: Standard z-score normalization
        self.X = self._standardize_features(self.X)
    
    def _standardize_features(self, X: np.ndarray) -> np.ndarray:
        # z = (x - μ) / σ for each feature
        # MATH VERIFIED: Correct standardization formula
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)
        stds[stds == 0] = 1.0  # Avoid division by zero
        
        X_standardized = (X - means) / stds  # Correct formula
        
        self.metadata['feature_means'] = means.tolist()
        self.metadata['feature_stds'] = stds.tolist()
        
        return X_standardized
    
    def _extract_metadata(self):
        self.metadata.update({
            'n_samples': len(self.X),
            'n_features': len(self.feature_names),
            'feature_names': self.feature_names,
            'positive_rate': float(np.mean(self.y)),
            'data_source': str(self.csv_path),
            'node_id': self.node_id
        })
    
    def get_full_dataset(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.X, self.y
    
    def get_batch(self, batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.X), size=batch_size, replace=False)
        return self.X[indices], self.y[indices]
    
    def get_statistics(self) -> Dict:
        # Statistical summary for anomaly detection
        return {
            'feature_names': self.feature_names,
            'feature_means': np.mean(self.X, axis=0).tolist(),
            'feature_stds': np.std(self.X, axis=0).tolist(),
            'feature_mins': np.min(self.X, axis=0).tolist(),
            'feature_maxs': np.max(self.X, axis=0).tolist(),
            'feature_medians': np.median(self.X, axis=0).tolist(),
            'feature_q25': np.percentile(self.X, 25, axis=0).tolist(),
            'feature_q75': np.percentile(self.X, 75, axis=0).tolist(),
            'positive_rate': float(np.mean(self.y)),
            'n_samples': len(self.X)
        }
    
    @staticmethod
    def generate_sample_csv(output_path: str, n_samples: int = 1000):
        logger = setup_logger('aegis.csv_generator')
        np.random.seed(42)
        
        # Generate realistic medical data
        patient_ids = [f"P{i:06d}" for i in range(n_samples)]
        ages = np.random.randint(18, 90, size=n_samples)
        genders = np.random.choice(['M', 'F'], size=n_samples)
        
        # Clinical features with realistic distributions
        systolic_bp = np.random.normal(120, 20, n_samples).clip(80, 200)
        diastolic_bp = np.random.normal(80, 10, n_samples).clip(50, 120)
        heart_rate = np.random.normal(75, 15, n_samples).clip(40, 150)
        glucose = np.random.normal(100, 25, n_samples).clip(60, 300)
        cholesterol = np.random.normal(200, 40, n_samples).clip(100, 400)
        bmi = np.random.normal(25, 5, n_samples).clip(15, 50)
        wbc_count = np.random.normal(7, 2, n_samples).clip(3, 15)
        hemoglobin = np.random.normal(14, 2, n_samples).clip(8, 20)
        
        # Diagnosis based on risk factors (realistic)
        risk_score = (
            (systolic_bp > 140).astype(int) +
            (glucose > 125).astype(int) +
            (cholesterol > 240).astype(int) +
            (bmi > 30).astype(int)
        )
        diagnosis = (risk_score >= 2).astype(int)
        
        df = pd.DataFrame({
            'patient_id': patient_ids,
            'age': ages,
            'gender': genders,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'bmi': bmi,
            'wbc_count': wbc_count,
            'hemoglobin': hemoglobin,
            'diagnosis': diagnosis
        })
        
        df.to_csv(output_path, index=False)
        logger.info(f"Generated CSV with {n_samples} records: {output_path}")
        return output_path
