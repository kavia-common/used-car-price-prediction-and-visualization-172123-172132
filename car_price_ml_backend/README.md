# Car Price ML Backend (Flask)

A Flask-based backend that trains and serves a machine learning model to predict used car prices. It exposes REST endpoints for prediction and model metrics, and generates OpenAPI docs for easy integration with the frontend.

- Default server port: 3001
- OpenAPI/Swagger UI: /docs (e.g., http://localhost:3001/docs)
- Health check: GET /

## Features
- Data-driven training using a CSV file (DATA_PATH).
- Preprocessing with OneHotEncoder for categorical features and passthrough for numeric features.
- Model choices: RandomForestRegressor (default) or LinearRegression.
- Model persistence to disk (MODEL_PATH) with accompanying metadata.
- REST API:
  - POST /predict for single-sample inference
  - GET /metrics for readiness and performance metrics
- CORS support controlled by ALLOWED_ORIGINS.

## Quick Start

1) Clone repository and enter backend folder:
- used-car-price-prediction-and-visualization-172123-172132/car_price_ml_backend

2) Create and activate a Python environment (recommended).

3) Install dependencies:
- pip install -r requirements.txt

4) Configure environment variables:
- Copy .env.example to .env and adjust as needed.

Environment variables:
- MODEL_PATH=models/model.joblib
- DATA_PATH=data/car_data.csv
- TRAIN_ON_START=true
- ALLOWED_ORIGINS=*
- MODEL_TYPE=random_forest

Notes:
- ALLOWED_ORIGINS can be a comma-separated list or "*".
- MODEL_TYPE accepts "rf", "random_forest", or "linear". Internally the app normalizes to rf or linear behaviors.

5) Provide training data:
- Place your car_data.csv at data/car_data.csv (create the data/ folder if it does not exist) or update DATA_PATH to point to your CSV.

6) Run the server:
- python run.py
- The API will be available at http://localhost:3001
- Swagger UI at http://localhost:3001/docs

If running in the provided environment, preview URLs may be mapped by the workspace. Ensure port 3001 is exposed and available.

## Training Flow and Model Readiness

- On startup, the backend reads .env and configures CORS and API metadata.
- If TRAIN_ON_START=true (or one of "1","true","yes","on") and DATA_PATH exists while MODEL_PATH does not exist:
  - The app will train a model at startup using the CSV at DATA_PATH.
  - Trained model and metadata are saved next to MODEL_PATH (metadata path is derived automatically).
- If TRAIN_ON_START=false (or not set), or the CSV is missing, training will be skipped.
- The model is considered "ready" when both model and metadata files are available on disk. GET /metrics reports readiness.

Manual training (optional):
- You can trigger training manually using the helper module:
- python -m app.ml.train --force
- This reads from DATA_PATH and writes to MODEL_PATH.

## Data Schema (Expected Columns)

The pipeline expects the following features and target in the dataset; it tolerates extra columns.

Categorical:
- brand
- model
- fuel_type
- transmission

Numeric:
- year (int)
- mileage_km (float)
- owner_count (int)
- engine_cc (float)
- seats (int)

Target:
- selling_price (float)

Missing values are handled with simple cleaning strategies; however, ensure your data is clean and representative for best results.

## API Reference

Base URL:
- http://localhost:3001

CORS:
- Controlled via ALLOWED_ORIGINS. Use "*" for development or specify exact origins (e.g., http://localhost:3000 for the frontend).

### Health
- GET /
- Response: {"message": "Healthy"}

### Metrics
- GET /metrics
- Purpose: Check whether a trained model is available and get evaluation metrics.
- Success 200 Response example:
{
  "ready": true,
  "metrics": { "r2": 0.82, "mae": 150000.5, "rmse": 220000.7 },
  "trained_at": 1736200000,
  "model_type": "rf",
  "features": ["brand","model","fuel_type","transmission","year","mileage_km","owner_count","engine_cc","seats"]
}
- If model not ready:
{
  "ready": false,
  "metrics": null,
  "trained_at": null,
  "model_type": null,
  "features": null
}

### Predict
- POST /predict
- Purpose: Predict a used car's price given its specifications.
- Request JSON body fields:
  - brand: string
  - model: string
  - fuel_type: string
  - transmission: string
  - year: integer
  - mileage_km: number
  - owner_count: integer
  - engine_cc: number
  - seats: integer

- Example request:
{
  "brand": "Toyota",
  "model": "Corolla",
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "year": 2018,
  "mileage_km": 45000,
  "owner_count": 1,
  "engine_cc": 1200,
  "seats": 5
}

- Success 200 Response:
{
  "predicted_price": 525000.0,
  "inputs": {
    "brand": "Toyota",
    "model": "Corolla",
    "fuel_type": "Petrol",
    "transmission": "Manual",
    "year": 2018,
    "mileage_km": 45000.0,
    "owner_count": 1,
    "engine_cc": 1200.0,
    "seats": 5
  },
  "model_info": {
    "model_type": "rf",
    "trained_at": 1736200000
  }
}

- Error 400: Invalid input or failed prediction.
- Error 503: MODEL_NOT_READY if the model or metadata are missing.
  - Example error payload:
  {
    "message": "Model not ready",
    "error": "MODEL_NOT_READY"
  }

## OpenAPI Generation

The project includes a helper script to regenerate the OpenAPI spec (interfaces/openapi.json) from the code and schemas:
- python generate_openapi.py

OpenAPI metadata is exposed under /docs for in-browser Swagger UI.

## Ports and Preview

- The backend listens on 0.0.0.0:3001 (see run.py).
- Ensure your environment or dev workspace forwards port 3001 so you can access:
  - API root: http://localhost:3001/
  - Swagger UI: http://localhost:3001/docs

## CORS

CORS is configured using the ALLOWED_ORIGINS environment variable:
- "*" allows all origins (useful for local development).
- For production, set a comma-separated whitelist, e.g.:
  - ALLOWED_ORIGINS=http://localhost:3000,https://your-frontend.app

## Optional Database

- This backend does not require a database to function. It reads from CSV and persists model artifacts to the filesystem.
- If a database is added later, update documentation and environment variables accordingly.

## Troubleshooting

- 503 MODEL_NOT_READY on /predict:
  - Ensure a model exists at MODEL_PATH with metadata. Either set TRAIN_ON_START=true and place DATA_PATH, or run:
    - python -m app.ml.train --force
- 400 on /predict:
  - Verify input schema matches required fields and value types.
- CORS errors in the browser:
  - Set ALLOWED_ORIGINS appropriately (e.g., include your frontend URL).

## License

This project is for demonstration and educational purposes.
