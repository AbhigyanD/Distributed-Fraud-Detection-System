"""Main orchestration script for the Distributed Fraud Detection System."""
import sys
import time
from typing import Dict, Any
from src.kafka.consumer import TransactionConsumer
from src.kafka.producer import AlertProducer
from src.pyspark.processor import TransactionProcessor
from src.models.ensemble import EnsembleFraudDetector
from src.models.mlflow_tracker import MLflowTracker
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class FraudDetectionPipeline:
    """Main fraud detection pipeline orchestrating all components."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the fraud detection pipeline."""
        if config is None:
            config = load_config()
        
        self.config = config
        
        # Initialize components
        logger.info("Initializing fraud detection pipeline components...")
        self.spark_processor = TransactionProcessor(config)
        self.ensemble_detector = EnsembleFraudDetector(config)
        self.alert_producer = AlertProducer(config)
        self.mlflow_tracker = MLflowTracker(config)
        
        logger.info("Fraud detection pipeline initialized")
    
    def process_transaction_batch(self, transactions: list[Dict[str, Any]]):
        """Process a batch of transactions through the pipeline."""
        try:
            logger.info(f"Processing batch of {len(transactions)} transactions")
            
            # Step 1: Process with PySpark
            processed_data = self.spark_processor.process_batch(transactions)
            processed_txns = processed_data['transactions']
            graph_data = processed_data['graph']
            
            # Step 2: Train models (if needed, can be done periodically)
            # For now, we'll assume models are pre-trained or train on first batch
            if not hasattr(self, 'models_trained'):
                logger.info("Training ensemble models on first batch...")
                self.ensemble_detector.train_models(
                    processed_txns,
                    graph_data
                )
                self.models_trained = True
            
            # Step 3: Detect fraud using ensemble
            predictions = self.ensemble_detector.predict_batch(
                processed_txns,
                graph_data
            )
            
            # Step 4: Reduce false positives
            filtered_predictions = self.ensemble_detector.reduce_false_positives(
                predictions,
                confidence_threshold=0.8
            )
            
            # Step 5: Send alerts for detected fraud
            fraud_count = 0
            for prediction in filtered_predictions:
                if prediction.get('is_fraud', False):
                    fraud_count += 1
                    alert = {
                        'transaction_id': prediction['transaction_id'],
                        'risk_score': prediction['weighted_score'],
                        'individual_scores': prediction['individual_scores'],
                        'timestamp': time.time(),
                        'alert_type': 'fraud_detected'
                    }
                    self.alert_producer.send_alert(alert)
            
            logger.info(f"Processed batch: {fraud_count} fraud alerts sent")
            
            # Step 6: Log metrics to MLflow
            if hasattr(self, 'mlflow_run'):
                self.mlflow_tracker.log_ensemble_metrics(filtered_predictions)
            
            return filtered_predictions
        
        except Exception as e:
            logger.error(f"Error processing transaction batch: {e}", exc_info=True)
            raise
    
    def start_streaming(self):
        """Start streaming fraud detection from Kafka."""
        logger.info("Starting fraud detection streaming pipeline...")
        
        # Start MLflow run
        self.mlflow_run = self.mlflow_tracker.start_run(run_name="fraud_detection_streaming")
        
        # Log configuration
        self.mlflow_tracker.log_params({
            'ensemble_models': ','.join(self.ensemble_detector.model_names),
            'voting_threshold': self.ensemble_detector.voting_threshold,
            'graph_embedding_dim': self.ensemble_detector.graphsage.embedding_dim if self.ensemble_detector.graphsage else 0
        })
        
        # Create consumer with message handler
        consumer = TransactionConsumer(
            config=self.config,
            message_handler=self.process_transaction_batch
        )
        
        try:
            # Start consuming
            consumer.consume_continuous()
        except KeyboardInterrupt:
            logger.info("Stopping fraud detection pipeline...")
        finally:
            consumer.close()
            self.alert_producer.close()
            self.spark_processor.stop()
            self.mlflow_tracker.end_run()
            logger.info("Fraud detection pipeline stopped")
    
    def process_historical_data(self, transactions: list[Dict[str, Any]]):
        """Process historical transaction data (batch mode)."""
        logger.info(f"Processing {len(transactions)} historical transactions")
        
        # Start MLflow run
        self.mlflow_run = self.mlflow_tracker.start_run(run_name="historical_fraud_detection")
        
        # Process in batches
        batch_size = self.config.get('kafka', {}).get('batch_size', 1000)
        all_predictions = []
        
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            predictions = self.process_transaction_batch(batch)
            all_predictions.extend(predictions)
        
        # Log final metrics
        self.mlflow_tracker.log_ensemble_metrics(all_predictions)
        
        # End run
        self.mlflow_tracker.end_run()
        
        return all_predictions


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Fraud Detection System')
    parser.add_argument(
        '--mode',
        choices=['streaming', 'batch'],
        default='streaming',
        help='Processing mode: streaming or batch'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(config)
    
    if args.mode == 'streaming':
        # Start streaming
        pipeline.start_streaming()
    else:
        # Batch mode - would need data source
        logger.info("Batch mode requires transaction data source")
        logger.info("Use streaming mode or provide data source for batch processing")


if __name__ == "__main__":
    main()

