"""
SentinelAI - Surveillance & CCTV Video Forensic Agent
Root Entrypoint for Streamlit Cloud & Local Deployment.
"""

import sys
from pathlib import Path

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Execute UI Application
from src.ui.app import main

if __name__ == "__main__":
    main()
