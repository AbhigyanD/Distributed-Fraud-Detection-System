"""PySpark processing pipeline for transaction data."""
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import (
    col, window, count, sum as spark_sum, avg, max as spark_max,
    min as spark_min, stddev, when, lit, udf, collect_list, hour, dayofweek
)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType, IntegerType
from typing import Dict, Any, List
from datetime import datetime
import json
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger(__name__)


class TransactionProcessor:
    """PySpark-based transaction processor for fraud detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Spark session and processor."""
        if config is None:
            config = load_config()
        
        spark_config = config.get('spark', {})
        processing_config = config.get('processing', {})
        
        # Build Spark session with Java security fix
        spark_builder = SparkSession.builder \
            .appName(spark_config.get('app_name', 'FraudDetectionSystem')) \
            .master(spark_config.get('master', 'local[*]')) \
            .config("spark.executor.memory", spark_config.get('executor_memory', '4g')) \
            .config("spark.driver.memory", spark_config.get('driver_memory', '2g')) \
            .config("spark.driver.maxResultSize", spark_config.get('max_result_size', '2g')) \
            .config("spark.sql.shuffle.partitions", spark_config.get('spark_sql_shuffle_partitions', 200)) \
            .config("spark.sql.streaming.checkpointLocation", "data/checkpoints") \
            .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
            .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        
        self.spark = spark_builder.getOrCreate()
        
        self.window_size = processing_config.get('window_size_seconds', 60)
        self.checkpoint_interval = processing_config.get('checkpoint_interval', '10 minutes')
        
        logger.info("Initialized Spark session for transaction processing")
    
    def get_transaction_schema(self) -> StructType:
        """Define schema for transaction data."""
        return StructType([
            StructField("transaction_id", StringType(), True),
            StructField("timestamp", TimestampType(), True),
            StructField("from_account", StringType(), True),
            StructField("to_account", StringType(), True),
            StructField("amount", DoubleType(), True),
            StructField("currency", StringType(), True),
            StructField("transaction_type", StringType(), True),
            StructField("merchant_id", StringType(), True),
            StructField("location", StringType(), True),
            StructField("device_id", StringType(), True),
            StructField("ip_address", StringType(), True),
            StructField("user_id", StringType(), True)
        ])
    
    def create_transaction_dataframe(self, transactions: List[Dict[str, Any]]) -> DataFrame:
        """Create Spark DataFrame from transaction list."""
        schema = self.get_transaction_schema()
        
        # Convert transactions to rows
        rows = []
        for txn in transactions:
            rows.append((
                txn.get('transaction_id'),
                datetime.fromisoformat(txn.get('timestamp', datetime.now().isoformat())),
                txn.get('from_account'),
                txn.get('to_account'),
                float(txn.get('amount', 0.0)),
                txn.get('currency', 'USD'),
                txn.get('transaction_type'),
                txn.get('merchant_id'),
                txn.get('location'),
                txn.get('device_id'),
                txn.get('ip_address'),
                txn.get('user_id')
            ))
        
        return self.spark.createDataFrame(rows, schema)
    
    def extract_features(self, df: DataFrame) -> DataFrame:
        """Extract features for fraud detection."""
        logger.info("Extracting features from transactions")
        
        # Time-based features
        df = df.withColumn("hour", hour(col("timestamp")))
        df = df.withColumn("day_of_week", dayofweek(col("timestamp")))
        
        # Amount-based features
        df = df.withColumn("amount_log", when(col("amount") > 0, 
                                               col("amount")).otherwise(0.001))
        df = df.withColumn("amount_log", col("amount_log").cast("double"))
        
        # Window aggregations per user
        window_spec = window(col("timestamp"), f"{self.window_size} seconds")
        
        # Add window column to original dataframe
        df = df.withColumn("window", window_spec)
        
        user_window_stats = df.groupBy("user_id", "window").agg(
            count("*").alias("txn_count_window"),
            spark_sum("amount").alias("total_amount_window"),
            avg("amount").alias("avg_amount_window"),
            spark_max("amount").alias("max_amount_window"),
            stddev("amount").alias("amount_stddev_window")
        )
        
        df = df.join(user_window_stats, ["user_id", "window"], "left")
        
        # Account-level aggregations
        from_account_stats = df.groupBy("from_account").agg(
            count("*").alias("from_account_txn_count"),
            avg("amount").alias("from_account_avg_amount"),
            stddev("amount").alias("from_account_amount_stddev")
        )
        
        df = df.join(from_account_stats, "from_account", "left")
        
        # To account stats
        to_account_stats = df.groupBy("to_account").agg(
            count("*").alias("to_account_txn_count"),
            avg("amount").alias("to_account_avg_amount")
        )
        
        df = df.join(to_account_stats, "to_account", "left")
        
        # Fill nulls
        df = df.fillna(0, subset=[
            "txn_count_window", "total_amount_window", "avg_amount_window",
            "max_amount_window", "amount_stddev_window",
            "from_account_txn_count", "from_account_avg_amount",
            "from_account_amount_stddev", "to_account_txn_count",
            "to_account_avg_amount"
        ])
        
        return df
    
    def build_transaction_graph(self, df: DataFrame) -> Dict[str, Any]:
        """Build graph structure from transactions for GraphSAGE."""
        logger.info("Building transaction graph")
        
        # Collect edges (transactions between accounts)
        edges_df = df.select(
            col("from_account").alias("src"),
            col("to_account").alias("dst"),
            col("amount"),
            col("timestamp"),
            col("transaction_id")
        ).distinct()
        
        # Collect nodes (unique accounts)
        from_nodes = df.select(col("from_account").alias("node_id")).distinct()
        to_nodes = df.select(col("to_account").alias("node_id")).distinct()
        nodes_df = from_nodes.union(to_nodes).distinct()
        
        # Convert to Python objects for graph construction
        edges = edges_df.collect()
        nodes = nodes_df.collect()
        
        graph_data = {
            "nodes": [row["node_id"] for row in nodes],
            "edges": [
                {
                    "src": row["src"],
                    "dst": row["dst"],
                    "amount": row["amount"],
                    "timestamp": str(row["timestamp"]),
                    "transaction_id": row["transaction_id"]
                }
                for row in edges
            ]
        }
        
        logger.info(f"Graph built: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")
        return graph_data
    
    def process_batch(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a batch of transactions."""
        logger.info(f"Processing batch of {len(transactions)} transactions")
        
        # Create DataFrame
        df = self.create_transaction_dataframe(transactions)
        
        # Extract features
        df_features = self.extract_features(df)
        
        # Build graph
        graph_data = self.build_transaction_graph(df)
        
        # Collect processed transactions
        processed_txns = df_features.collect()
        
        result = {
            "transactions": [
                {
                    "transaction_id": row["transaction_id"],
                    "features": {
                        "amount": row["amount"],
                        "hour": row["hour"],
                        "day_of_week": row["day_of_week"],
                        "txn_count_window": row["txn_count_window"],
                        "total_amount_window": row["total_amount_window"],
                        "avg_amount_window": row["avg_amount_window"],
                        "max_amount_window": row["max_amount_window"],
                        "amount_stddev_window": row["amount_stddev_window"],
                        "from_account_txn_count": row["from_account_txn_count"],
                        "to_account_txn_count": row["to_account_txn_count"]
                    }
                }
                for row in processed_txns
            ],
            "graph": graph_data
        }
        
        return result
    
    def stop(self):
        """Stop Spark session."""
        self.spark.stop()
        logger.info("Spark session stopped")

