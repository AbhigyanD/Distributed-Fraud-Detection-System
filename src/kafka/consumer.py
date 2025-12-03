"""Kafka consumer for processing transaction streams."""
import json
from typing import Dict, Any, Callable, Optional
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class TransactionConsumer:
    """Consumer for processing transaction streams from Kafka."""
    
    def __init__(
        self,
        config: Dict[str, Any] = None,
        message_handler: Optional[Callable] = None
    ):
        """Initialize Kafka consumer."""
        if config is None:
            config = load_config()
        
        kafka_config = config.get('kafka', {})
        self.bootstrap_servers = kafka_config.get('bootstrap_servers', 'localhost:9092')
        self.topic = kafka_config.get('topic_transactions', 'transactions')
        self.consumer_group = kafka_config.get('consumer_group', 'fraud_detection_group')
        self.batch_size = kafka_config.get('batch_size', 1000)
        self.message_handler = message_handler
        
        self.consumer = KafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.consumer_group,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset=kafka_config.get('auto_offset_reset', 'earliest'),
            enable_auto_commit=True,
            max_poll_records=self.batch_size
        )
        
        logger.info(f"Initialized Kafka consumer for topic: {self.topic}")
    
    def consume_messages(self, timeout_ms: int = 1000):
        """Consume messages from Kafka topic."""
        try:
            messages = self.consumer.poll(timeout_ms=timeout_ms)
            
            for topic_partition, records in messages.items():
                batch = []
                for record in records:
                    transaction = record.value
                    batch.append(transaction)
                
                if batch and self.message_handler:
                    self.message_handler(batch)
                
                logger.debug(f"Processed {len(batch)} messages from {topic_partition}")
        
        except KafkaError as e:
            logger.error(f"Error consuming messages: {e}")
            raise
    
    def consume_continuous(self):
        """Continuously consume messages."""
        logger.info("Starting continuous message consumption...")
        try:
            while True:
                self.consume_messages()
        except KeyboardInterrupt:
            logger.info("Stopping consumer...")
        finally:
            self.close()
    
    def close(self):
        """Close the consumer."""
        self.consumer.close()
        logger.info("Kafka consumer closed")

