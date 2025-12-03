#!/bin/bash
cd /Users/bu/IdeaProjects/Distributed-Fraud-Detection-System
source venv/bin/activate
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python scripts/test_system.py

