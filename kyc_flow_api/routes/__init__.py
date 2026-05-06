"""Routes package for KYC Flow API."""

from .ocr_routes import register_ocr_routes
from .liveness_routes import register_liveness_routes


def register_all_routes(app):
    """Register all routes to Flask app."""
    register_ocr_routes(app)
    register_liveness_routes(app)


__all__ = [
    'register_all_routes',
    'register_ocr_routes',
    'register_liveness_routes',
]
