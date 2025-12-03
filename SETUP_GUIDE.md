# Setup Guide - Distributed Fraud Detection System

This guide will walk you through setting up and running the Distributed Fraud Detection System step by step.

## Prerequisites Checklist

- [ ] Python 3.8 or higher installed
- [ ] Java 8 or higher installed (required for PySpark)
- [ ] At least 8GB RAM available
- [ ] Apache Kafka installed and configured
- [ ] Internet connection for downloading dependencies

## Step-by-Step Setup

### Step 1: Install Java

**macOS (using Homebrew):**
```bash
brew install openjdk@11
export JAVA_HOME=$(/usr/libexec/java_home -v 11)
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get update
sudo apt-get install openjdk-11-jdk
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
```

**Windows:**
- Download Java from https://adoptium.net/
- Set JAVA_HOME environment variable

Verify installation:
```bash
java -version
```

### Step 2: Install Apache Kafka

**Download Kafka:**
```bash
# Download from https://kafka.apache.org/downloads
# Extract the archive
tar -xzf kafka_2.13-3.6.0.tgz
cd kafka_2.13-3.6.0
```

**Start Zookeeper:**
```bash
bin/zookeeper-server-start.sh config/zookeeper.properties
```

**Start Kafka (in a new terminal):**
```bash
cd kafka_2.13-3.6.0
bin/kafka-server-start.sh config/server.properties
```

**Create Topics:**
```bash
# In a third terminal
cd kafka_2.13-3.6.0
bin/kafka-topics.sh --create --topic transactions --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1
bin/kafka-topics.sh --create --topic fraud_alerts --bootstrap-server localhost:9092 --partitions 3 --replication-factor 1

# Verify topics were created
bin/kafka-topics.sh --list --bootstrap-server localhost:9092
```

### Step 3: Set Up Python Environment

**Navigate to project directory:**
```bash
cd Distributed-Fraud-Detection-System
```

**Create virtual environment:**
```bash
python3 -m venv venv
```

**Activate virtual environment:**
```bash
# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

**Upgrade pip:**
```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

**Install Python packages:**
```bash
pip install -r requirements.txt
```

**Note:** This may take several minutes. If you encounter issues:

- **PySpark issues**: Ensure Java is properly installed and JAVA_HOME is set
- **PyTorch Geometric issues**: May need to install PyTorch first separately
- **Memory issues**: Install packages one at a time if needed

### Step 5: Verify Installation

**Test imports:**
```bash
python -c "import pyspark; import kafka; import mlflow; import torch; print('All imports successful!')"
```

### Step 6: Configure the System

**Review configuration:**
```bash
cat config/config.yaml
```

**Edit if needed:**
- Adjust Kafka bootstrap servers if not using localhost:9092
- Modify Spark memory settings based on your system
- Tune model hyperparameters as needed

### Step 7: Test the System

**Terminal 1 - Start Fraud Detection Pipeline:**
```bash
python -m src.main --mode streaming
```

**Terminal 2 - Generate Sample Data:**
```bash
python scripts/generate_sample_data.py --count 1000 --fraud-rate 0.1 --delay 0.01
```

**Terminal 3 - Monitor Alerts:**
```bash
cd kafka_2.13-3.6.0
bin/kafka-console-consumer.sh --topic fraud_alerts --from-beginning --bootstrap-server localhost:9092
```

You should see:
- Transactions being processed in Terminal 1
- Sample data being generated in Terminal 2
- Fraud alerts appearing in Terminal 3

### Step 8: View MLflow Experiments

**Start MLflow UI:**
```bash
mlflow ui --backend-store-uri file:./mlruns
```

**Open in browser:**
```
http://localhost:5000
```

## Common Issues and Solutions

### Issue: "Java not found"
**Solution:** Set JAVA_HOME environment variable
```bash
export JAVA_HOME=/path/to/java
```

### Issue: "Kafka connection refused"
**Solution:** 
- Ensure Kafka is running
- Check if port 9092 is available
- Verify bootstrap servers in config

### Issue: "Out of memory errors"
**Solution:**
- Reduce Spark executor/driver memory in `config/config.yaml`
- Reduce batch size
- Close other applications

### Issue: "PyTorch Geometric installation fails"
**Solution:**
```bash
pip install torch torchvision torchaudio
pip install torch-geometric
```

### Issue: "Module not found errors"
**Solution:**
- Ensure virtual environment is activated
- Reinstall requirements: `pip install -r requirements.txt`
- Check Python path

## Performance Tuning

### For High-Throughput (1M+ transactions/hour)

1. **Increase Spark resources:**
```yaml
spark:
  executor_memory: "8g"
  driver_memory: "4g"
  spark_sql_shuffle_partitions: 400
```

2. **Optimize Kafka:**
- Increase partitions: `bin/kafka-topics.sh --alter --topic transactions --partitions 10 --bootstrap-server localhost:9092`
- Adjust batch size in config

3. **Use GPU for GraphSAGE:**
- Install CUDA-enabled PyTorch
- System will automatically use GPU if available

## Next Steps

1. **Customize Models:**
   - Adjust ensemble weights in `config/config.yaml`
   - Tune GraphSAGE hyperparameters
   - Modify feature engineering in `src/pyspark/processor.py`

2. **Add Real Data:**
   - Replace sample data generator with real Kafka producer
   - Adjust schema in `src/pyspark/processor.py` if needed

3. **Production Deployment:**
   - Set up proper Kafka cluster
   - Use Spark cluster instead of local mode
   - Configure MLflow with database backend
   - Add monitoring and alerting

## Getting Help

- Check logs in console output
- Review MLflow experiments for model performance
- Check Kafka topics for message flow
- Review configuration files

## System Requirements Summary

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Disk | 10GB free | 50GB+ free |
| Java | 8+ | 11+ |
| Python | 3.8+ | 3.10+ |

---

**Ready to detect fraud!** 🚀

