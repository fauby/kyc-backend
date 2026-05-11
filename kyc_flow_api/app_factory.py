"""FastAPI application factory for KYC Flow API."""

from fastapi import FastAPI
from .routes import register_all_routes


app = FastAPI(title="KYC Flow API")
register_all_routes(app)
