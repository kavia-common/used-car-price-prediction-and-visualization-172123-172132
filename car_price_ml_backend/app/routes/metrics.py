from __future__ import annotations

from typing import Any, Dict, Optional

from flask_smorest import Blueprint
from flask.views import MethodView

from app.schemas import MetricsResponseSchema
from app.ml.pipeline import load_model_and_meta

blp = Blueprint(
    "Metrics",
    "metrics",
    url_prefix="/metrics",
    description="Endpoints to fetch model readiness and evaluation metrics.",
)


@blp.route("")
class MetricsView(MethodView):
    # PUBLIC_INTERFACE
    @blp.response(200, MetricsResponseSchema)
    def get(self):
        """
        Get model readiness and metrics.

        Returns:
            MetricsResponseSchema:
                - ready: bool
                - metrics: r2, mae, rmse when available
                - trained_at: unix timestamp when trained
                - model_type: rf/linear/etc.
                - features: list of feature names used in training
        """
        model, meta = load_model_and_meta()

        if model is None or meta is None:
            # Model not present or metadata missing -> not ready
            return {
                "ready": False,
                "metrics": None,
                "trained_at": None,
                "model_type": None,
                "features": None,
            }

        metrics: Optional[Dict[str, Any]] = meta.get("metrics") if isinstance(meta, dict) else None
        features = []
        feat = (meta or {}).get("features", {})
        if isinstance(feat, dict):
            # Concatenate categorical + numeric for the flat "features" in schema
            features = (feat.get("categorical") or []) + (feat.get("numeric") or [])

        return {
            "ready": True,
            "metrics": metrics,
            "trained_at": meta.get("trained_at"),
            "model_type": meta.get("model_type"),
            "features": features,
        }
