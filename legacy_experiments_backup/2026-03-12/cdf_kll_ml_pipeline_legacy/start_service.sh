#!/bin/bash
# Start OASIS KLL correction model service

cd "$(dirname "$0")"

echo "Starting OASIS model service on http://localhost:8080"
echo "Endpoints:"
echo "  POST /predict - Correct KLL histograms"
echo "  GET /health - Health check"
echo ""

python3 model_service.py
