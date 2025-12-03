"""MLflow integration for model tracking and versioning."""
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from typing import Dict, Any, Optional
import os
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class MLflowTracker:
    """MLflow tracker for experiment tracking and model registry."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize MLflow tracker."""
        if config is None:
            config = load_config()
        
        mlflow_config = config.get('mlflow', {})
        tracking_uri = mlflow_config.get('tracking_uri', 'file:./mlruns')
        experiment_name = mlflow_config.get('experiment_name', 'fraud_detection')
        
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        self.model_registry = mlflow_config.get('model_registry', 'fraud_models')
        
        logger.info(f"Initialized MLflow tracker: experiment={experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Start a new MLflow run."""
        run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run.info.run_id}")
        return run
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow."""
        mlflow.log_params(params)
        logger.debug(f"Logged parameters: {params}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow."""
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged metrics: {metrics}")
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        model_type: str = "pytorch"
    ):
        """Log model to MLflow."""
        if model_type == "pytorch":
            mlflow.pytorch.log_model(model, artifact_path)
        elif model_type == "sklearn":
            mlflow.sklearn.log_model(model, artifact_path)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        logger.info(f"Logged {model_type} model to {artifact_path}")
    
    def log_graphsage_model(self, graphsage_model, artifact_path: str = "graphsage_model"):
        """Log GraphSAGE model to MLflow."""
        if graphsage_model.model is not None:
            mlflow.pytorch.log_model(
                graphsage_model.model,
                artifact_path,
                registered_model_name=f"{self.model_registry}_graphsage"
            )
            logger.info(f"Logged GraphSAGE model to {artifact_path}")
    
    def log_ensemble_metrics(
        self,
        predictions: list,
        true_labels: Optional[list] = None
    ):
        """Log ensemble model performance metrics."""
        if true_labels is None:
            # Calculate metrics without ground truth
            fraud_count = sum(1 for p in predictions if p.get('is_fraud', False))
            avg_score = sum(p.get('weighted_score', 0.0) for p in predictions) / len(predictions)
            
            metrics = {
                'fraud_detected_count': fraud_count,
                'fraud_rate': fraud_count / len(predictions) if predictions else 0.0,
                'avg_risk_score': avg_score
            }
        else:
            # Calculate metrics with ground truth
            tp = sum(1 for p, label in zip(predictions, true_labels) 
                    if p.get('is_fraud', False) and label == 1)
            fp = sum(1 for p, label in zip(predictions, true_labels) 
                    if p.get('is_fraud', False) and label == 0)
            fn = sum(1 for p, label in zip(predictions, true_labels) 
                    if p.get('is_fraud', False) == False and label == 1)
            tn = sum(1 for p, label in zip(predictions, true_labels) 
                    if p.get('is_fraud', False) == False and label == 0)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            metrics = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'accuracy': accuracy,
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        
        self.log_metrics(metrics)
        return metrics
    
    def log_false_positive_reduction(
        self,
        before_count: int,
        after_count: int
    ):
        """Log false positive reduction metrics."""
        reduction = ((before_count - after_count) / before_count * 100) if before_count > 0 else 0.0
        
        metrics = {
            'false_positives_before': before_count,
            'false_positives_after': after_count,
            'false_positive_reduction_percent': reduction
        }
        
        self.log_metrics(metrics)
        logger.info(f"False positive reduction: {reduction:.1f}%")
        return metrics
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
        logger.info("Ended MLflow run")

