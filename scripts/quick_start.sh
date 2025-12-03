#!/bin/bash

# Quick Start Script for Distributed Fraud Detection System
# This script helps you get started quickly

echo "========================================="
echo "Distributed Fraud Detection System"
echo "Quick Start Script"
echo "========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import pyspark" 2>/dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    echo "Dependencies already installed."
fi

# Check if Kafka topics exist
echo ""
echo "Checking Kafka setup..."
if command -v kafka-topics.sh &> /dev/null; then
    echo "Kafka found. Checking topics..."
    kafka-topics.sh --list --bootstrap-server localhost:9092 &> /dev/null
    if [ $? -eq 0 ]; then
        echo "Kafka is running!"
    else
        echo "WARNING: Kafka might not be running. Please start Kafka first."
        echo "  Start Zookeeper: bin/zookeeper-server-start.sh config/zookeeper.properties"
        echo "  Start Kafka: bin/kafka-server-start.sh config/server.properties"
    fi
else
    echo "WARNING: Kafka not found in PATH. Please ensure Kafka is installed and configured."
fi

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To start the fraud detection system:"
echo "  1. Terminal 1: python -m src.main --mode streaming"
echo "  2. Terminal 2: python scripts/generate_sample_data.py --count 1000"
echo "  3. Terminal 3: kafka-console-consumer.sh --topic fraud_alerts --from-beginning --bootstrap-server localhost:9092"
echo ""
echo "To view MLflow experiments:"
echo "  mlflow ui --backend-store-uri file:./mlruns"
echo ""

