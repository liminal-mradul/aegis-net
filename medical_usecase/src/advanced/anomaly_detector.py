import numpy as np
from typing import Dict, Tuple, Optional
from ..utils.logger import setup_logger

class AnomalyDetector:
    """
    Anomaly detection for federated learning with robust error handling.
    Detects poisoning attacks and data quality issues.
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.logger = setup_logger('aegis.anomaly_detector')
        self.network_stats: Optional[Dict] = None
        self.local_stats: Optional[Dict] = None
    
    def set_network_statistics(self, stats: Dict):
        """Update network reference statistics with validation"""
        # Ensure all required fields are present
        required_fields = ['feature_means', 'feature_stds']
        missing = [f for f in required_fields if f not in stats]
        
        if missing:
            self.logger.warning(f"Network stats missing fields: {missing}")
            return
        
        self.network_stats = stats
        self.logger.info("Updated network reference statistics")
    
    def set_local_statistics(self, stats: Dict):
        """Update local statistics"""
        self.local_stats = stats
        self.logger.info("Updated local statistics")
    
    def detect_zscore_anomalies(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-score method: z = |x - μ| / σ
        Flag if z > threshold (typically 3)
        """
        if self.network_stats is None or 'feature_means' not in self.network_stats:
            self.logger.debug("No network stats available, using local statistics")
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
        else:
            means = np.array(self.network_stats['feature_means'])
            stds = np.array(self.network_stats['feature_stds'])
        
        # Compute Z-scores: z = |x - μ| / σ
        z_scores = np.abs((data - means) / (stds + 1e-8))
        
        # Anomaly if ANY feature exceeds threshold
        anomaly_mask = np.any(z_scores > threshold, axis=1)
        
        n_anomalies = np.sum(anomaly_mask)
        self.logger.debug(f"Z-score: {n_anomalies}/{len(data)} anomalies ({100*n_anomalies/len(data):.2f}%)")
        
        return anomaly_mask, z_scores
    
    def detect_iqr_anomalies(self, data: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        IQR method: outliers outside [Q1 - k*IQR, Q3 + k*IQR]
        k = 1.5 for outliers, 3.0 for extreme outliers
        """
        # Check if we have network stats with quartiles
        if (self.network_stats is not None and 
            'feature_q25' in self.network_stats and 
            'feature_q75' in self.network_stats):
            q25 = np.array(self.network_stats['feature_q25'])
            q75 = np.array(self.network_stats['feature_q75'])
        else:
            # Compute from local data
            self.logger.debug("Computing IQR from local data (no network quartiles)")
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)
        
        # IQR = Q3 - Q1
        iqr = q75 - q25
        
        # Bounds: [Q1 - k*IQR, Q3 + k*IQR]
        lower_bound = q25 - multiplier * iqr
        upper_bound = q75 + multiplier * iqr
        
        # Outlier score: how far outside bounds
        outlier_scores = np.maximum(
            lower_bound - data,  # Below lower bound
            data - upper_bound   # Above upper bound
        )
        outlier_scores = np.maximum(outlier_scores, 0)  # Only positive
        
        anomaly_mask = np.any(outlier_scores > 0, axis=1)
        
        n_anomalies = np.sum(anomaly_mask)
        self.logger.debug(f"IQR: {n_anomalies}/{len(data)} anomalies ({100*n_anomalies/len(data):.2f}%)")
        
        return anomaly_mask, outlier_scores
    
    def detect_distribution_shift(self) -> Dict:
        """Compare local vs network distribution"""
        if self.network_stats is None or self.local_stats is None:
            return {'status': 'insufficient_data', 'feature_shifts': []}
        
        if 'feature_names' not in self.local_stats:
            return {'status': 'missing_feature_names', 'feature_shifts': []}
        
        results = {'feature_shifts': [], 'significant_shifts': []}
        feature_names = self.local_stats.get('feature_names', [])
        
        # Ensure we have the required fields
        required = ['feature_means', 'feature_stds']
        if not all(f in self.local_stats for f in required):
            return {'status': 'missing_local_stats', 'feature_shifts': []}
        if not all(f in self.network_stats for f in required):
            return {'status': 'missing_network_stats', 'feature_shifts': []}
        
        for i in range(len(feature_names)):
            if i >= len(self.local_stats['feature_means']):
                break
            
            local_mean = self.local_stats['feature_means'][i]
            local_std = self.local_stats['feature_stds'][i]
            network_mean = self.network_stats['feature_means'][i]
            network_std = self.network_stats['feature_stds'][i]
            
            # Normalized mean difference
            mean_diff = abs(local_mean - network_mean) / (network_std + 1e-8)
            
            # Std ratio (log scale for symmetry)
            std_ratio = local_std / (network_std + 1e-8)
            
            # Combined shift score
            shift_score = mean_diff + abs(np.log(std_ratio + 1e-8))
            
            result = {
                'feature': feature_names[i] if i < len(feature_names) else f'feature_{i}',
                'local_mean': float(local_mean),
                'network_mean': float(network_mean),
                'mean_difference': float(mean_diff),
                'std_ratio': float(std_ratio),
                'shift_score': float(shift_score),
                'significant': shift_score > 2.0
            }
            
            results['feature_shifts'].append(result)
            if result['significant']:
                results['significant_shifts'].append(feature_names[i])
        
        n_sig = len(results['significant_shifts'])
        results['summary'] = f"{n_sig}/{len(feature_names)} features shifted"
        results['status'] = 'complete'
        
        if n_sig > 0:
            self.logger.warning(f"Distribution shift: {results['significant_shifts']}")
        
        return results
    
    def detect_label_imbalance(self) -> Dict:
        """Test if local positive rate differs from network"""
        if self.network_stats is None or self.local_stats is None:
            return {'status': 'insufficient_data'}
        
        if 'positive_rate' not in self.local_stats or 'positive_rate' not in self.network_stats:
            return {'status': 'missing_positive_rate'}
        
        local_rate = self.local_stats.get('positive_rate', 0)
        network_rate = self.network_stats.get('positive_rate', 0)
        n_local = self.local_stats.get('n_samples', 0)
        
        if n_local == 0:
            return {'status': 'no_data'}
        
        # Standard error for proportion test: SE = sqrt(p(1-p)/n)
        se = np.sqrt(network_rate * (1 - network_rate) / n_local + 1e-8)
        
        # Z-score: z = (p1 - p2) / SE
        z_score = abs(local_rate - network_rate) / (se + 1e-8)
        
        # Two-tailed test: p < 0.05 ⟹ z > 1.96
        is_anomalous = z_score > 1.96
        
        result = {
            'local_positive_rate': float(local_rate),
            'network_positive_rate': float(network_rate),
            'difference': float(abs(local_rate - network_rate)),
            'z_score': float(z_score),
            'is_anomalous': bool(is_anomalous),
            'severity': 'high' if z_score > 3 else 'medium' if z_score > 2 else 'low',
            'status': 'complete'
        }
        
        if is_anomalous:
            self.logger.warning(
                f"Label imbalance: local={local_rate:.3f}, "
                f"network={network_rate:.3f}, z={z_score:.2f}"
            )
        
        return result
    
    def comprehensive_analysis(self, data: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Run comprehensive anomaly analysis with proper error handling.
        
        Args:
            data: Feature matrix (n_samples, n_features)
            labels: Label array (n_samples,)
        
        Returns:
            Dictionary with analysis results
        """
        self.logger.info("Running comprehensive anomaly analysis")
        
        results = {
            'node_id': self.node_id,
            'n_samples': len(data),
            'analyses': {}
        }
        
        try:
            # Z-score anomalies
            zscore_mask, zscore_vals = self.detect_zscore_anomalies(data)
            results['analyses']['zscore'] = {
                'n_anomalies': int(np.sum(zscore_mask)),
                'anomaly_rate': float(np.mean(zscore_mask)),
                'max_zscore': float(np.max(zscore_vals))
            }
        except Exception as e:
            self.logger.error(f"Z-score analysis failed: {e}")
            results['analyses']['zscore'] = {'status': 'error', 'message': str(e)}
        
        try:
            # IQR anomalies
            iqr_mask, iqr_scores = self.detect_iqr_anomalies(data)
            results['analyses']['iqr'] = {
                'n_anomalies': int(np.sum(iqr_mask)),
                'anomaly_rate': float(np.mean(iqr_mask)),
                'max_outlier_score': float(np.max(iqr_scores))
            }
        except Exception as e:
            self.logger.error(f"IQR analysis failed: {e}")
            results['analyses']['iqr'] = {'status': 'error', 'message': str(e)}
        
        # Distribution shift (only if network stats available)
        if self.network_stats is not None:
            try:
                shift_results = self.detect_distribution_shift()
                results['analyses']['distribution_shift'] = shift_results
            except Exception as e:
                self.logger.error(f"Distribution shift analysis failed: {e}")
                results['analyses']['distribution_shift'] = {'status': 'error', 'message': str(e)}
        
        # Label imbalance (only if network stats available)
        if self.network_stats is not None:
            try:
                imbalance_results = self.detect_label_imbalance()
                results['analyses']['label_imbalance'] = imbalance_results
            except Exception as e:
                self.logger.error(f"Label imbalance analysis failed: {e}")
                results['analyses']['label_imbalance'] = {'status': 'error', 'message': str(e)}
        
        # Overall assessment
        try:
            total_anomalies = 0
            if 'zscore' in results['analyses'] and 'n_anomalies' in results['analyses']['zscore']:
                total_anomalies += results['analyses']['zscore']['n_anomalies']
            if 'iqr' in results['analyses'] and 'n_anomalies' in results['analyses']['iqr']:
                total_anomalies += results['analyses']['iqr']['n_anomalies']
            
            anomaly_rate = total_anomalies / (2 * len(data)) if len(data) > 0 else 0
            
            results['overall'] = {
                'total_anomalies': int(total_anomalies),
                'overall_anomaly_rate': float(anomaly_rate),
                'status': 'critical' if anomaly_rate > 0.15 else 
                         'high' if anomaly_rate > 0.10 else 
                         'medium' if anomaly_rate > 0.05 else 'low'
            }
        except Exception as e:
            self.logger.error(f"Overall assessment failed: {e}")
            results['overall'] = {
                'status': 'error',
                'message': str(e)
            }
        
        self.logger.info(f"Analysis complete: status={results['overall'].get('status', 'unknown')}")
        
        return results
    
    def get_anomalous_samples(self, data: np.ndarray, method: str = 'zscore', 
                            **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get anomalous samples using specified method.
        
        Args:
            data: Feature matrix
            method: 'zscore' or 'iqr'
            **kwargs: Additional parameters for detection method
        
        Returns:
            (anomaly_indices, anomalous_samples) tuple
        """
        if method == 'zscore':
            mask, _ = self.detect_zscore_anomalies(data, **kwargs)
        elif method == 'iqr':
            mask, _ = self.detect_iqr_anomalies(data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'")
        
        anomaly_indices = np.where(mask)[0]
        return anomaly_indices, data[mask]
