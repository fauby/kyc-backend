"""Routes package for KYC Flow API."""

from .ocr_routes import router as ocr_router
from .liveness_routes import router as liveness_router


def register_all_routes(app):
    """Register all routes to FastAPI app."""
    app.include_router(ocr_router)
    app.include_router(liveness_router)
