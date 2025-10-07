from __future__ import annotations

from typing import Any, Dict

from flask_smorest import Blueprint, abort
from flask.views import MethodView

from app.schemas import PredictRequestSchema, PredictResponseSchema
from app.ml.pipeline import predict_single, load_model_and_meta, ModelNotReadyError

blp = Blueprint(
    "Predict",
    "predict",
    url_prefix="/predict",
    description="Endpoints for price prediction.",
)


@blp.route("")
class PredictView(MethodView):
    # PUBLIC_INTERFACE
    @blp.arguments(PredictRequestSchema, location="json")
    @blp.response(200, PredictResponseSchema)
    def post(self, json_data: Dict[str, Any]):
        """
        Predict selling price for a single car specification.

        Request body:
            PredictRequestSchema: Validated input features.

        Returns:
            PredictResponseSchema: Predicted price with echoed sanitized inputs and model info.

        Error responses:
            400: Validation errors (handled by flask-smorest).
            503: MODEL_NOT_READY if model or metadata is not available.
        """
        # Ensure model and metadata are available for prediction
        model, meta = load_model_and_meta()
        if model is None or meta is None:
            abort(
                503,
                message="Model not ready",
                error="MODEL_NOT_READY",
            )

        try:
            predicted_price = predict_single(json_data)
        except ModelNotReadyError:
            abort(
                503,
                message="Model not ready",
                error="MODEL_NOT_READY",
            )
        except Exception as e:
            # Return a concise error to the client
            abort(400, message="Unable to generate prediction", error=str(e))

        # Prepare response aligned with PredictResponseSchema
        model_info = {
            "model_type": meta.get("model_type", "unknown"),
            "trained_at": meta.get("trained_at", 0),
        }

        return {
            "predicted_price": float(predicted_price),
            "inputs": json_data,
            "model_info": model_info,
        }
