"""Direct Flask launcher for the KYC Flow API."""

import os
import sys


ROOT_DIR = os.path.dirname(__file__)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kyc_flow_api.app_factory import app


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
