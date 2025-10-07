"""
Command-style entry points to train the ML model using the pipeline utilities.

This module intentionally avoids running training on import. It exposes a main()
that can be called programmatically or via "python -m app.ml.train" to perform
training if data is available.

Environment variables:
- See app.ml.pipeline for DATA_PATH, MODEL_PATH, TRAIN_ON_START, MODEL_TYPE.
"""

from __future__ import annotations

import argparse
from typing import Optional

from .pipeline import (
    get_default_paths,
    train_model,
    train_if_needed,
    _read_csv_safely,  # type: ignore
)


# PUBLIC_INTERFACE
def main(force: bool = False) -> Optional[dict]:
    """Train a model from DATA_PATH and persist artifacts. Returns metadata or None if skipped."""
    paths = get_default_paths()
    if force:
        df = _read_csv_safely(paths.data_path)
        if df is None:
            print(f"[train] No data found at {paths.data_path}. Skipping.")
            return None
        meta = train_model(df=df, paths=paths)
        print(f"[train] Trained and saved model to {paths.model_path}")
        return meta
    else:
        meta = train_if_needed(force=False)
        if meta is None:
            print("[train] Training skipped (model exists, TRAIN_ON_START not set, or no data).")
        else:
            print(f"[train] Trained and saved model to {paths.model_path}")
        return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train car price model.")
    parser.add_argument("--force", action="store_true", help="Force retraining even if model exists.")
    args = parser.parse_args()
    main(force=args.force)
