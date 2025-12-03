"""Kafka producer for streaming transaction data."""
import json
import time
from typing import Dict, Any
from kafka import KafkaProducer
from kafka.errors import KafkaError
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class TransactionProducer:
    """Producer for streaming transaction data to Kafka."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Kafka producer."""
        if config is None:
            config = load_config()
        
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.topic = kafka_config.get('topic_transactions', 'transactions')
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            key_serializer=lambda k: k.encode('utf-8') if k else None,
            acks='all',
            retries=3,
            max_in_flight_requests_per_connection=1
        )
        
        logger.info(f"Initialized Kafka producer for topic: {self.topic}")
    
    def send_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Send a single transaction to Kafka."""
        try:
            # Use transaction ID as key for partitioning
            key = transaction.get('transaction_id', None)
            future = self.producer.send(self.topic, key=key, value=transaction)
            
            # Wait for the message to be sent
            record_metadata = future.get(timeout=10)
            logger.debug(
                f"Transaction sent: topic={record_metadata.topic}, "
                f"partition={record_metadata.partition}, offset={record_metadata.offset}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to send transaction: {e}")
            return False
    
    def send_batch(self, transactions: list[Dict[str, Any]]) -> int:
        """Send a batch of transactions."""
        success_count = 0
        for transaction in transactions:
            if self.send_transaction(transaction):
                success_count += 1
        return success_count
    
    def close(self):
        """Close the producer."""
        self.producer.close()
        logger.info("Kafka producer closed")


class AlertProducer:
    """Producer for sending fraud alerts to Kafka."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize alert producer."""
        if config is None:
            config = load_config()
        
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.topic = kafka_config.get('topic_alerts', 'fraud_alerts')
        
        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        
        logger.info(f"Initialized alert producer for topic: {self.topic}")
    
    def send_alert(self, alert: Dict[str, Any]) -> bool:
        """Send a fraud alert to Kafka."""
        try:
            future = self.producer.send(self.topic, value=alert)
            record_metadata = future.get(timeout=10)
            logger.info(
                f"Alert sent: transaction_id={alert.get('transaction_id')}, "
                f"risk_score={alert.get('risk_score')}"
            )
            return True
        except KafkaError as e:
            logger.error(f"Failed to send alert: {e}")
            return False
    
    def close(self):
        """Close the producer."""
        self.producer.close()

