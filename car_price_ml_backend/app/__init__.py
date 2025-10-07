import os
from typing import Optional

from flask import Flask
from flask_cors import CORS

# PUBLIC_INTERFACE
def create_app() -> Flask:
    """Factory to create and configure the Flask application.

    - Loads environment variables from a .env file if present.
    - Configures CORS using ALLOWED_ORIGINS env (default '*').
    - Sets up OpenAPI docs via flask-smorest if available.
    - Registers health, predict, and metrics blueprints.
    - Keeps the app resilient if optional blueprints or model files are absent.
    """
    # Lazy import to keep dependencies minimal at import time
    try:
        from dotenv import load_dotenv  # optional; only affects local dev if available
        load_dotenv()
    except Exception:
        # It's okay if python-dotenv isn't installed; envs may be provided at runtime
        pass

    app = Flask(__name__)
    app.url_map.strict_slashes = False

    # Configure CORS from env var; default to "*"
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
    CORS(app, resources={r"/*": {"origins": allowed_origins}})

    # If flask-smorest is available (it is in requirements), set up OpenAPI
    from flask_smorest import Api
    app.config["API_TITLE"] = os.getenv("API_TITLE", "Car Price ML API")
    app.config["API_VERSION"] = os.getenv("API_VERSION", "v1")
    app.config["OPENAPI_VERSION"] = "3.0.3"
    app.config["OPENAPI_URL_PREFIX"] = "/docs"
    app.config["OPENAPI_SWAGGER_UI_PATH"] = ""
    app.config["OPENAPI_SWAGGER_UI_URL"] = "https://cdn.jsdelivr.net/npm/swagger-ui-dist/"
    api = Api(app)

    # Register existing health blueprint
    try:
        from .routes.health import blp as health_blp
        api.register_blueprint(health_blp)
    except Exception as e:
        # Do not fail startup if health route import fails
        app.logger.warning(f"Health blueprint not registered: {e}")

    # Prepare to optionally register predict and metrics blueprints if/when created
    for dotted_path in [
        "app.routes.predict:blp",
        "app.routes.metrics:blp",
    ]:
        try:
            module_name, attr = dotted_path.split(":")
            mod = __import__(module_name, fromlist=[attr])
            candidate = getattr(mod, attr, None)
            if candidate is not None:
                api.register_blueprint(candidate)
        except Exception:
            # Silently skip until those routes exist
            continue

    # Expose api on app for tools like generate_openapi.py
    app.extensions = getattr(app, "extensions", {})
    app.extensions["smorest_api"] = api  # convenience handle

    # Attach for backward compatibility with generate_openapi.py which imports `app` and `api`
    global api_instance
    api_instance = api

    return app


# Create a module-level app instance for compatibility with existing imports
app = create_app()

# Backward compatibility for generate_openapi.py which imports `api` from app
# Provide the Api instance used above without re-importing the class to avoid linter conflicts
try:
    api = app.extensions.get("smorest_api")  # type: ignore[assignment]
except Exception:  # pragma: no cover
    api = None  # type: ignore[assignment]
