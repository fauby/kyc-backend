"""Flask application factory for KYC Flow API."""

from flask import Flask
from .routes import register_all_routes
from .config import FLASK_HOST, OCR_PORT


def create_app():
    """
    Create and configure Flask application.
    
    Returns:
        Flask app instance with all routes registered
    """
    app = Flask(__name__)
    
    # Register all routes
    register_all_routes(app)
    
    return app


app = create_app()


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=OCR_PORT, debug=False)
