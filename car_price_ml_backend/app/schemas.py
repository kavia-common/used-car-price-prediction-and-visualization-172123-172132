"""
Marshmallow schemas for request/response validation and OpenAPI for the Car Price ML API.

These schemas are designed to be used with flask-smorest (@blp.arguments / @blp.response)
and align with the ML pipeline's expected features defined in app.ml.pipeline.

Notes:
- Feature names are consistent with the pipeline: categorical (brand, model, fuel_type, transmission)
  and numeric (year, mileage_km, owner_count, engine_cc, seats).
- Validation includes simple bounds and presence checks where appropriate.
- Response schemas include structured metadata to aid clients.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from marshmallow import Schema, fields, validate, validates_schema, ValidationError


def _string_field(description: str, allow_unknown_placeholder: bool = True) -> fields.String:
    """Helper to create a trimmed string field with basic validation."""
    validators = [validate.Length(min=1, error="Must not be empty.")]
    # Allow "Unknown" as a valid categorical placeholder used in pipeline preprocessing.
    if allow_unknown_placeholder:
        # Custom validator to allow "Unknown" literal besides non-empty strings
        def _non_empty_or_unknown(value: str) -> None:
            if value is None:
                raise ValidationError("Value is required.")
            v = str(value).strip()
            if not v and v != "Unknown":
                raise ValidationError("Must not be empty.")
        validators = [_non_empty_or_unknown]
    return fields.String(
        required=True,
        allow_none=False,
        description=description,
        validate=validators,
        error_messages={"required": "This field is required.", "null": "This field cannot be null."},
    )


def _float_field(description: str, minimum: Optional[float] = None, maximum: Optional[float] = None) -> fields.Float:
    """Helper to create a float field with range validators and consistent errors."""
    validators = []
    if minimum is not None:
        validators.append(validate.Range(min=minimum, error=f"Must be greater than or equal to {minimum}"))
    if maximum is not None:
        validators.append(validate.Range(max=maximum, error=f"Must be less than or equal to {maximum}"))
    return fields.Float(
        required=True,
        allow_none=False,
        validate=validators,
        description=description,
        error_messages={
            "required": "This field is required.",
            "invalid": "Invalid number.",
            "null": "This field cannot be null.",
        },
    )


def _int_field(description: str, minimum: Optional[int] = None, maximum: Optional[int] = None) -> fields.Integer:
    """Helper to create an integer field with range validators and consistent errors."""
    validators = []
    if minimum is not None:
        validators.append(validate.Range(min=minimum, error=f"Must be greater than or equal to {minimum}"))
    if maximum is not None:
        validators.append(validate.Range(max=maximum, error=f"Must be less than or equal to {maximum}"))
    return fields.Integer(
        required=True,
        allow_none=False,
        validate=validators,
        description=description,
        error_messages={
            "required": "This field is required.",
            "invalid": "Invalid integer.",
            "null": "This field cannot be null.",
        },
    )


class _ModelInfoSchema(Schema):
    """Schema describing basic model information returned with responses."""
    model_type = fields.String(
        required=True,
        description="Type of the trained model, e.g., 'rf' for RandomForest or 'linear' for Linear Regression.",
    )
    trained_at = fields.Integer(
        required=True,
        description="Unix timestamp (seconds since epoch) when the model was trained.",
    )


class _MetricsSchema(Schema):
    """Schema representing evaluation metrics."""
    r2 = fields.Float(required=True, description="R-squared score.")
    mae = fields.Float(required=True, description="Mean Absolute Error.")
    rmse = fields.Float(required=True, description="Root Mean Squared Error.")


# PUBLIC_INTERFACE
class PredictRequestSchema(Schema):
    """Schema for prediction input payload aligned with pipeline features."""

    # Categorical features
    brand = _string_field("Brand of the car (categorical).")
    model = _string_field("Model of the car (categorical).")
    fuel_type = _string_field("Fuel type (e.g., Petrol, Diesel, Electric).")
    transmission = _string_field("Transmission type (e.g., Manual, Automatic).")

    # Numeric features
    year = _int_field("Manufacturing year of the car.", minimum=1950, maximum=datetime.now().year + 1)
    mileage_km = _float_field("Total mileage (kilometers driven).", minimum=0)
    owner_count = _int_field("Number of previous owners.", minimum=0, maximum=10)
    engine_cc = _float_field("Engine displacement in cubic centimeters (cc).", minimum=100)
    seats = _int_field("Number of seats.", minimum=1, maximum=20)

    @validates_schema
    def _cross_field_validation(self, data: Dict[str, Any], **kwargs: Any) -> None:
        """Cross-field validations for more meaningful error messages."""
        # Example: future years are already constrained; ensure realistic engine sizes with seats
        engine = data.get("engine_cc")
        seats = data.get("seats")
        if engine is not None and seats is not None:
            if seats >= 7 and engine < 800:
                raise ValidationError(
                    {"engine_cc": "For vehicles with 7 or more seats, engine_cc is expected to be >= 800 cc."}
                )


# PUBLIC_INTERFACE
class PredictResponseSchema(Schema):
    """Schema for prediction response."""

    predicted_price = fields.Float(
        required=True,
        description="Predicted selling price for the given input."
    )
    inputs = fields.Dict(
        keys=fields.String(),
        values=fields.Raw(),
        required=True,
        description="Echo of the sanitized inputs used for prediction."
    )
    model_info = fields.Nested(
        _ModelInfoSchema,
        required=True,
        description="Information about the model used for the prediction."
    )


# PUBLIC_INTERFACE
class MetricsResponseSchema(Schema):
    """Schema for model readiness and metrics response."""

    ready = fields.Boolean(
        required=True,
        description="Indicates if a trained model and metadata are available."
    )
    metrics = fields.Nested(
        _MetricsSchema,
        required=False,
        allow_none=True,
        description="Evaluation metrics (present only when ready is true)."
    )
    trained_at = fields.Integer(
        required=False,
        allow_none=True,
        description="Unix timestamp when model was last trained (present only when ready is true)."
    )
    model_type = fields.String(
        required=False,
        allow_none=True,
        description="Type of the trained model, e.g., 'rf' or 'linear' (present only when ready is true)."
    )
    features = fields.List(
        fields.String(),
        required=False,
        allow_none=True,
        description="List of feature names used during training."
    )


# Export commonly used schema instances for convenience with flask-smorest
# Example usage:
#   @blp.arguments(predict_request)
#   @blp.response(200, predict_response)
predict_request = PredictRequestSchema()
predict_response = PredictResponseSchema()
metrics_response = MetricsResponseSchema()
