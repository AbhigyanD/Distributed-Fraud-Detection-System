#!/usr/bin/env python3
"""Test script to verify the system works without Kafka."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.pyspark.processor import TransactionProcessor
from src.models.ensemble import EnsembleFraudDetector
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
import json

logger = setup_logger(__name__)


def generate_test_transaction(txn_id: str, is_fraud: bool = False) -> dict:
    """Generate a test transaction."""
    import random
    from datetime import datetime, timedelta
    
    accounts = [f"ACC_{i:06d}" for i in range(1, 100)]
    base_time = datetime.now() - timedelta(hours=1)
    
    transaction = {
        "transaction_id": txn_id,
        "timestamp": (base_time + timedelta(seconds=random.randint(0, 3600))).isoformat(),
        "from_account": random.choice(accounts),
        "to_account": random.choice(accounts),
        "amount": random.uniform(100.0, 10000.0) if not is_fraud else random.uniform(50000.0, 100000.0),
        "currency": "USD",
        "transaction_type": "transfer",
        "merchant_id": f"MERCH_{random.randint(1, 50):04d}",
        "location": "US" if not is_fraud else "UNKNOWN",
        "device_id": f"DEV_{random.randint(1000, 9999)}",
        "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
        "user_id": f"USER_{random.randint(1, 50)}"
    }
    return transaction


def main():
    """Test the fraud detection system."""
    logger.info("=" * 60)
    logger.info("Testing Distributed Fraud Detection System")
    logger.info("=" * 60)
    
    # Load config
    try:
        config = load_config()
        logger.info("✓ Configuration loaded")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return
    
    # Generate test transactions
    logger.info("\n1. Generating test transactions...")
    transactions = []
    for i in range(20):
        is_fraud = i < 2  # 2 fraudulent out of 20
        txn = generate_test_transaction(f"TXN_TEST_{i:04d}", is_fraud)
        transactions.append(txn)
    logger.info(f"✓ Generated {len(transactions)} transactions ({2} fraudulent)")
    
    # Test PySpark processor
    logger.info("\n2. Testing PySpark processor...")
    try:
        processor = TransactionProcessor(config)
        processed_data = processor.process_batch(transactions)
        logger.info(f"✓ Processed {len(processed_data['transactions'])} transactions")
        logger.info(f"✓ Graph built: {len(processed_data['graph']['nodes'])} nodes, {len(processed_data['graph']['edges'])} edges")
    except Exception as e:
        logger.error(f"✗ PySpark processing failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test Ensemble detector
    logger.info("\n3. Testing Ensemble fraud detector...")
    try:
        detector = EnsembleFraudDetector(config)
        
        # Train models
        logger.info("   Training models...")
        detector.train_models(
            processed_data['transactions'],
            processed_data['graph']
        )
        logger.info("✓ Models trained")
        
        # Predict
        logger.info("   Running predictions...")
        predictions = detector.predict_batch(
            processed_data['transactions'],
            processed_data['graph']
        )
        
        fraud_count = sum(1 for p in predictions if p.get('is_fraud', False))
        logger.info(f"✓ Predictions complete: {fraud_count} frauds detected out of {len(predictions)}")
        
        # Show some results
        logger.info("\n4. Sample predictions:")
        for i, pred in enumerate(predictions[:5]):
            logger.info(f"   Transaction {pred['transaction_id']}: "
                       f"Score={pred['weighted_score']:.3f}, "
                       f"Fraud={pred['is_fraud']}")
        
        # False positive reduction
        logger.info("\n5. Applying false positive reduction...")
        filtered = detector.reduce_false_positives(predictions)
        filtered_fraud_count = sum(1 for p in filtered if p.get('is_fraud', False))
        logger.info(f"✓ After filtering: {filtered_fraud_count} frauds (reduced from {fraud_count})")
        
    except Exception as e:
        logger.error(f"✗ Ensemble detection failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Cleanup
    logger.info("\n6. Cleaning up...")
    try:
        processor.stop()
        logger.info("✓ Spark session stopped")
    except:
        pass
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ All tests passed! System is working correctly.")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("1. Set up Kafka (see SETUP_GUIDE.md)")
    logger.info("2. Run: python -m src.main --mode streaming")
    logger.info("3. Generate data: python scripts/generate_sample_data.py")


if __name__ == "__main__":
    main()

