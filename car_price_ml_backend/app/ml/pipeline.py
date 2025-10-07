"""
ML pipeline and training utilities for used car price prediction.

This module provides functions to:
- Load training data from a CSV at DATA_PATH.
- Build a scikit-learn Pipeline with preprocessing (OneHotEncoder for categorical columns)
  and a regressor (RandomForestRegressor by default, LinearRegression if MODEL_TYPE=linear).
- Train/evaluate with train/test split and compute R2, MAE, RMSE.
- Persist the trained pipeline to MODEL_PATH using joblib, and write a metadata.json with
  metrics and timestamps.
- Load model and metadata for inference.
- Helper to perform single-sample prediction from a dict payload.
- Guarded training on startup with TRAIN_ON_START env; training only happens when explicitly
  invoked by train_if_needed().

Environment variables:
- DATA_PATH: path to CSV data file (default: data/car_data.csv)
- MODEL_PATH: path to persist the trained model (default: models/model.joblib)
- MODEL_TYPE: "rf" (RandomForestRegressor) or "linear" (LinearRegression). Default: "rf"
- TRAIN_ON_START: "1" or "true" to attempt training on startup via train_if_needed(); default "0"

The pipeline tolerates extra columns in the data and safely drops unknowns in OneHotEncoder.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# -----------------------------
# Defaults and configuration
# -----------------------------

DEFAULT_DATA_PATH = "data/car_data.csv"
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_METADATA_PATH = "models/metadata.json"

# Canonical feature set; tolerate extra columns by selecting intersection later.
CATEGORICAL_FEATURES = [
    "brand",
    "model",
    "fuel_type",
    "transmission",
]
NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "owner_count",
    "engine_cc",
    "seats",
]
TARGET_COLUMN = "selling_price"


class ModelNotReadyError(RuntimeError):
    """Raised when prediction is requested but the model or metadata is not yet available."""
    pass


@dataclass
class Paths:
    """Dataclass for resolved file paths for model and metadata."""
    model_path: str
    metadata_path: str
    data_path: str


# PUBLIC_INTERFACE
def get_default_paths() -> Paths:
    """Resolve model, metadata, and data paths from environment variables with sensible defaults."""
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    data_path = os.getenv("DATA_PATH", DEFAULT_DATA_PATH)
    # metadata is colocated next to model by default
    default_meta = DEFAULT_METADATA_PATH if DEFAULT_METADATA_PATH.startswith("models/") else "metadata.json"
    if model_path.endswith(".joblib"):
        metadata_path = model_path.replace(".joblib", ".metadata.json")
    else:
        # fallback
        metadata_path = os.getenv("METADATA_PATH", default_meta)
    return Paths(model_path=model_path, metadata_path=metadata_path, data_path=data_path)


def _ensure_dir(path: str) -> None:
    """Ensure the directory for the specified file path exists."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def _get_model_type() -> str:
    """Return normalized model type from env."""
    return os.getenv("MODEL_TYPE", "rf").strip().lower()


def _build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    """Create a ColumnTransformer with OneHotEncoder for categoricals and passthrough for numerics."""
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_features),
            ("num", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _build_estimator(model_type: str) -> Any:
    """Return a regressor instance based on model_type."""
    if model_type == "linear":
        return LinearRegression()
    # default to RandomForestRegressor with small footprint
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )


def _build_pipeline(model_type: str, categorical_features: list[str], numeric_features: list[str]) -> Pipeline:
    """Construct the full sklearn Pipeline."""
    preprocessor = _build_preprocessor(categorical_features, numeric_features)
    estimator = _build_estimator(model_type)
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", estimator)])
    return pipe


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: drop rows with missing target, coerce numeric fields, and drop rows with completely missing features."""
    # Drop rows without target
    if TARGET_COLUMN in df.columns:
        df = df.dropna(subset=[TARGET_COLUMN])

    # Coerce numeric columns; non-convertible become NaN
    for col in NUMERIC_FEATURES + [c for c in [TARGET_COLUMN] if c in NUMERIC_FEATURES]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Optional: fill NA for numerics with median to avoid dropping many rows
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            median = df[col].median()
            df[col] = df[col].fillna(median)

    # For categoricals, replace NaN with "Unknown"
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str)

    return df


def _select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Select intersection of required features, tolerate extra columns, and return X, y."""
    # Proceed even if some features are missing; let model train on available subset where possible
    cat_present = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    num_present = [c for c in NUMERIC_FEATURES if c in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data.")

    X = df[cat_present + num_present].copy()
    y = df[TARGET_COLUMN].copy()

    # Remove rows where all selected features are NA/empty
    if not X.empty:
        non_empty_mask = ~(X.isna() | (X.apply(lambda s: s.astype(str).str.strip() == "", axis=0))).all(axis=1)
        X = X[non_empty_mask]
        y = y.loc[X.index]

    return X, y


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"r2": float(r2), "mae": float(mae), "rmse": float(rmse)}


def _read_csv_safely(path: str) -> Optional[pd.DataFrame]:
    """Read CSV if present; return None if missing or unreadable."""
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


# PUBLIC_INTERFACE
def load_model_and_meta(paths: Optional[Paths] = None) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    """Load a persisted model Pipeline and metadata if they exist.

    Returns:
        (model, metadata) where either may be None if not found.
    """
    p = paths or get_default_paths()
    model, metadata = None, None
    try:
        if os.path.exists(p.model_path):
            model = joblib.load(p.model_path)
    except Exception:
        model = None
    try:
        if os.path.exists(p.metadata_path):
            with open(p.metadata_path, "r") as f:
                metadata = json.load(f)
    except Exception:
        metadata = None
    return model, metadata


def _persist(model: Pipeline, metadata: Dict[str, Any], paths: Paths) -> None:
    """Persist the trained model and metadata to disk."""
    _ensure_dir(paths.model_path)
    _ensure_dir(paths.metadata_path)
    joblib.dump(model, paths.model_path)
    with open(paths.metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


# PUBLIC_INTERFACE
def train_model(
    df: pd.DataFrame,
    model_type: Optional[str] = None,
    paths: Optional[Paths] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Train a model on provided DataFrame and persist model and metadata.

    Args:
        df: Data containing features and target.
        model_type: "rf" or "linear"; defaults from env.
        paths: Optional Paths override; defaults from env.
        test_size: Proportion for test split.
        random_state: Seed for reproducibility.

    Returns:
        metadata dict with metrics, feature info, and file paths.
    """
    model_type = (model_type or _get_model_type()).lower()
    p = paths or get_default_paths()

    df = _clean_dataframe(df)
    X, y = _select_feature_columns(df)

    # Determine present feature sets
    cat_present = [c for c in CATEGORICAL_FEATURES if c in X.columns]
    num_present = [c for c in NUMERIC_FEATURES if c in X.columns]

    if X.empty or y.empty:
        raise ValueError("Insufficient data after cleaning to train the model.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    pipe = _build_pipeline(model_type=model_type, categorical_features=cat_present, numeric_features=num_present)
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    metrics = _evaluate(y_test.values if hasattr(y_test, "values") else y_test, preds)

    metadata: Dict[str, Any] = {
        "model_type": model_type,
        "trained_at": int(time.time()),
        "features": {
            "categorical": cat_present,
            "numeric": num_present,
            "target": TARGET_COLUMN,
        },
        "metrics": metrics,
        "paths": {
            "model_path": p.model_path,
            "metadata_path": p.metadata_path,
            "data_path": p.data_path,
        },
        "train_test": {
            "test_size": test_size,
            "random_state": random_state,
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
        },
    }

    _persist(pipe, metadata, p)
    return metadata


# PUBLIC_INTERFACE
def train_if_needed(force: bool = False) -> Optional[Dict[str, Any]]:
    """Train a model if not present or if TRAIN_ON_START indicates to do so.

    Behavior:
    - If a model exists and force is False, do nothing and return None.
    - If TRAIN_ON_START env is truthy ("1", "true", "yes") attempt to train if data exists.
    - If force=True, training occurs regardless of existing model.
    """
    p = get_default_paths()
    truthy = {"1", "true", "yes", "on"}
    should_train = force or os.getenv("TRAIN_ON_START", "0").strip().lower() in truthy

    model_exists = os.path.exists(p.model_path)
    if model_exists and not force:
        return None

    if not should_train and not force:
        return None

    df = _read_csv_safely(p.data_path)
    if df is None:
        # Gracefully skip training if no data is available
        return None

    return train_model(df=df, paths=p)


# PUBLIC_INTERFACE
def predict_single(payload: Dict[str, Any], paths: Optional[Paths] = None) -> float:
    """Predict selling price for a single payload dictionary.

    Args:
        payload: dict with keys matching feature names (extra keys tolerated).
        paths: optional override of model/metadata paths.

    Returns:
        Predicted selling price (float).

    Raises:
        ModelNotReadyError: if model is missing.
        ValueError: if payload cannot form a valid input row.
    """
    p = paths or get_default_paths()
    model, metadata = load_model_and_meta(p)
    if model is None or metadata is None:
        raise ModelNotReadyError("Model not trained or not available. Please train first.")

    # Build a single-row DataFrame using the features used during training
    feat = metadata.get("features", {})
    cat = feat.get("categorical", [])
    num = feat.get("numeric", [])

    # tolerate extra keys; select only needed ones, fill missing with defaults
    row: Dict[str, Any] = {}

    for c in cat:
        val = payload.get(c, "Unknown")
        row[c] = "Unknown" if val in (None, "", "NaN") else str(val)

    for n in num:
        val = payload.get(n, np.nan)
        try:
            row[n] = float(val)
        except (TypeError, ValueError):
            row[n] = np.nan

    X = pd.DataFrame([row], columns=cat + num)

    # Numeric missing -> fill with 0 or simple heuristic (could be improved)
    for n in num:
        if X[n].isna().any():
            X[n] = X[n].fillna(0.0)

    pred = model.predict(X)
    # prediction returns array([value])
    return float(pred[0])


# Convenience: do not trigger training at import time; only provide helpers.
# End of module.
