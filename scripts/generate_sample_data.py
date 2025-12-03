"""Script to generate sample transaction data for testing."""
import json
import random
import time
from datetime import datetime, timedelta
from src.kafka.producer import TransactionProducer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def generate_sample_transaction(transaction_id: str) -> dict:
    """Generate a sample transaction."""
    accounts = [f"ACC_{i:06d}" for i in range(1, 1000)]
    merchants = [f"MERCH_{i:04d}" for i in range(1, 100)]
    locations = ["US", "UK", "CA", "AU", "DE", "FR", "IT", "ES"]
    currencies = ["USD", "EUR", "GBP", "CAD", "AUD"]
    transaction_types = ["purchase", "transfer", "withdrawal", "deposit"]
    
    # Generate transaction
    base_time = datetime.now() - timedelta(days=1)
    timestamp = base_time + timedelta(seconds=random.randint(0, 86400))
    
    transaction = {
        "transaction_id": transaction_id,
        "timestamp": timestamp.isoformat(),
        "from_account": random.choice(accounts),
        "to_account": random.choice(accounts),
        "amount": round(random.uniform(10.0, 10000.0), 2),
        "currency": random.choice(currencies),
        "transaction_type": random.choice(transaction_types),
        "merchant_id": random.choice(merchants),
        "location": random.choice(locations),
        "device_id": f"DEV_{random.randint(1000, 9999)}",
        "ip_address": f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}",
        "user_id": f"USER_{random.randint(1, 500)}"
    }
    
    return transaction


def generate_fraudulent_transaction(transaction_id: str) -> dict:
    """Generate a suspicious/fraudulent transaction."""
    transaction = generate_sample_transaction(transaction_id)
    
    # Make it suspicious
    transaction["amount"] = round(random.uniform(5000.0, 50000.0), 2)  # Large amount
    transaction["location"] = "UNKNOWN"  # Suspicious location
    
    return transaction


def main():
    """Generate and send sample transactions."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample transaction data')
    parser.add_argument('--count', type=int, default=1000, help='Number of transactions to generate')
    parser.add_argument('--fraud-rate', type=float, default=0.1, help='Fraud rate (0.0 to 1.0)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between transactions (seconds)')
    
    args = parser.parse_args()
    
    producer = TransactionProducer()
    
    fraud_count = int(args.count * args.fraud_rate)
    normal_count = args.count - fraud_count
    
    logger.info(f"Generating {args.count} transactions ({fraud_count} fraudulent, {normal_count} normal)")
    
    try:
        for i in range(args.count):
            transaction_id = f"TXN_{int(time.time() * 1000)}_{i:06d}"
            
            if i < fraud_count:
                transaction = generate_fraudulent_transaction(transaction_id)
            else:
                transaction = generate_sample_transaction(transaction_id)
            
            producer.send_transaction(transaction)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Sent {i + 1} transactions")
            
            time.sleep(args.delay)
    
    except KeyboardInterrupt:
        logger.info("Stopped generating transactions")
    finally:
        producer.close()
        logger.info("Sample data generation complete")


if __name__ == "__main__":
    main()

