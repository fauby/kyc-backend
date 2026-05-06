"""KYC Flow API - Main Flask application entry point."""

from app_factory import app

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
