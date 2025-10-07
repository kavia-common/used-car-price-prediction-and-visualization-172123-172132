#!/bin/bash
cd /home/kavia/workspace/code-generation/used-car-price-prediction-and-visualization-172123-172132/car_price_ml_backend
source venv/bin/activate
flake8 .
LINT_EXIT_CODE=$?
if [ $LINT_EXIT_CODE -ne 0 ]; then
  exit 1
fi

