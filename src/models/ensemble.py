"""Ensemble methods for fraud detection combining multiple models."""
import numpy as np
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest
import xgboost as xgb
from src.models.graphsage import GraphSAGEModel
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class IsolationForestModel:
    """Isolation Forest model for anomaly detection."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        """Initialize Isolation Forest."""
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        self.is_trained = False
    
    def train(self, features: np.ndarray):
        """Train the Isolation Forest model."""
        self.model.fit(features)
        self.is_trained = True
        logger.info("Isolation Forest model trained")
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Isolation Forest returns -1 for anomalies, 1 for normal
        predictions = self.model.predict(features)
        # Convert to scores (0-1 scale, higher = more anomalous)
        scores = (1 - predictions) / 2  # -1 -> 1.0, 1 -> 0.0
        return scores


class XGBoostModel:
    """XGBoost model for fraud detection."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        """Initialize XGBoost model."""
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        self.is_trained = False
    
    def train(self, features: np.ndarray, labels: np.ndarray = None):
        """Train XGBoost model."""
        if labels is None:
            # Unsupervised: use all as negative class (0)
            labels = np.zeros(len(features))
        
        # XGBoost classifier needs at least some positive examples
        # If all labels are 0, create a simple binary classification
        # by using feature-based anomaly scores as pseudo-labels
        if np.all(labels == 0) and len(labels) > 0:
            # Create pseudo-labels based on feature outliers
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            # Use distance from mean as anomaly score
            mean_dist = np.linalg.norm(features_scaled - np.mean(features_scaled, axis=0), axis=1)
            # Top 10% as positive class
            threshold = np.percentile(mean_dist, 90)
            labels = (mean_dist > threshold).astype(int)
            # Ensure at least one positive example
            if np.sum(labels) == 0:
                labels[np.argmax(mean_dist)] = 1
        
        self.model.fit(features, labels)
        self.is_trained = True
        logger.info("XGBoost model trained")
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict fraud probability."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Get probability of positive class (fraud)
        proba = self.model.predict_proba(features)
        if proba.shape[1] == 2:
            return proba[:, 1]  # Return fraud probability
        else:
            return proba[:, 0]


class EnsembleFraudDetector:
    """Ensemble fraud detector combining multiple models."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize ensemble detector."""
        if config is None:
            config = load_config()
        
        ensemble_config = config.get('ensemble', {})
        self.model_names = ensemble_config.get('models', ['graphsage', 'xgboost', 'isolation_forest'])
        self.weights = ensemble_config.get('weights', {
            'graphsage': 0.5,
            'xgboost': 0.3,
            'isolation_forest': 0.2
        })
        self.voting_threshold = ensemble_config.get('voting_threshold', 0.6)
        
        # Initialize models
        self.graphsage = GraphSAGEModel(config) if 'graphsage' in self.model_names else None
        self.xgboost = XGBoostModel() if 'xgboost' in self.model_names else None
        self.isolation_forest = IsolationForestModel() if 'isolation_forest' in self.model_names else None
        
        logger.info(f"Initialized ensemble with models: {self.model_names}")
    
    def extract_features_for_ml(self, transactions: List[Dict[str, Any]]) -> np.ndarray:
        """Extract features for traditional ML models."""
        features = []
        
        for txn in transactions:
            txn_features = txn.get('features', {})
            feature_vector = [
                txn_features.get('amount', 0.0),
                txn_features.get('hour', 0),
                txn_features.get('day_of_week', 0),
                txn_features.get('txn_count_window', 0),
                txn_features.get('total_amount_window', 0.0),
                txn_features.get('avg_amount_window', 0.0),
                txn_features.get('max_amount_window', 0.0),
                txn_features.get('amount_stddev_window', 0.0),
                txn_features.get('from_account_txn_count', 0),
                txn_features.get('to_account_txn_count', 0)
            ]
            features.append(feature_vector)
        
        return np.array(features)
    
    def train_models(
        self,
        transactions: List[Dict[str, Any]],
        graph_data: Dict[str, Any] = None,
        labels: np.ndarray = None
    ):
        """Train all ensemble models."""
        logger.info("Training ensemble models")
        
        # Extract features
        features = self.extract_features_for_ml(transactions)
        
        # Train XGBoost
        if self.xgboost:
            self.xgboost.train(features, labels)
        
        # Train Isolation Forest
        if self.isolation_forest:
            self.isolation_forest.train(features)
        
        # Train GraphSAGE (if graph data provided)
        if self.graphsage and graph_data:
            try:
                data, node_ids = self.graphsage.build_graph_data(graph_data)
                self.graphsage.initialize_model()
                self.graphsage.train_model(data, node_ids=node_ids)
            except Exception as e:
                logger.warning(f"GraphSAGE training failed: {e}")
    
    def predict_single_transaction(
        self,
        transaction: Dict[str, Any],
        graph_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Predict fraud score for a single transaction."""
        scores = {}
        weighted_score = 0.0
        
        # Extract features
        features = self.extract_features_for_ml([transaction])
        
        # XGBoost prediction
        if self.xgboost and self.xgboost.is_trained:
            xgb_score = float(self.xgboost.predict_proba(features)[0])
            scores['xgboost'] = xgb_score
            weighted_score += xgb_score * self.weights.get('xgboost', 0.0)
        
        # Isolation Forest prediction
        if self.isolation_forest and self.isolation_forest.is_trained:
            if_score = float(self.isolation_forest.predict(features)[0])
            scores['isolation_forest'] = if_score
            weighted_score += if_score * self.weights.get('isolation_forest', 0.0)
        
        # GraphSAGE prediction
        if self.graphsage and graph_data:
            try:
                data, node_ids = self.graphsage.build_graph_data(graph_data)
                node_scores = self.graphsage.predict(data, node_ids=node_ids)
                # Use average node score or specific node score
                graph_score = float(np.mean(list(node_scores.values())))
                scores['graphsage'] = graph_score
                weighted_score += graph_score * self.weights.get('graphsage', 0.0)
            except Exception as e:
                logger.warning(f"GraphSAGE prediction failed: {e}")
        
        # Final decision
        is_fraud = weighted_score >= self.voting_threshold
        
        return {
            'transaction_id': transaction.get('transaction_id'),
            'weighted_score': weighted_score,
            'individual_scores': scores,
            'is_fraud': is_fraud,
            'threshold': self.voting_threshold
        }
    
    def predict_batch(
        self,
        transactions: List[Dict[str, Any]],
        graph_data: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Predict fraud scores for a batch of transactions."""
        logger.info(f"Predicting fraud for {len(transactions)} transactions")
        
        results = []
        
        for transaction in transactions:
            result = self.predict_single_transaction(transaction, graph_data)
            results.append(result)
        
        # Count frauds
        fraud_count = sum(1 for r in results if r['is_fraud'])
        logger.info(f"Detected {fraud_count} fraudulent transactions out of {len(results)}")
        
        return results
    
    def reduce_false_positives(
        self,
        predictions: List[Dict[str, Any]],
        confidence_threshold: float = 0.8
    ) -> List[Dict[str, Any]]:
        """Reduce false positives using confidence-based filtering."""
        filtered = []
        
        for pred in predictions:
            # Only flag as fraud if weighted score is high AND at least one model is very confident
            individual_scores = pred.get('individual_scores', {})
            max_individual_score = max(individual_scores.values()) if individual_scores else 0.0
            
            # Require both high weighted score and high individual confidence
            if pred['is_fraud']:
                if pred['weighted_score'] >= self.voting_threshold and max_individual_score >= confidence_threshold:
                    filtered.append(pred)
                else:
                    # Reduce false positive
                    pred['is_fraud'] = False
                    pred['false_positive_reduced'] = True
                    filtered.append(pred)
            else:
                filtered.append(pred)
        
        original_frauds = sum(1 for p in predictions if p['is_fraud'])
        filtered_frauds = sum(1 for p in filtered if p['is_fraud'])
        reduction = ((original_frauds - filtered_frauds) / original_frauds * 100) if original_frauds > 0 else 0
        
        logger.info(f"False positive reduction: {reduction:.1f}% ({original_frauds} -> {filtered_frauds})")
        
        return filtered

