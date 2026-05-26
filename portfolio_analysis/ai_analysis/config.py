"""
Configuration settings for the AI Analysis commentary pass.
Handles API credentials, model parameters, timeouts, and boundaries.
"""

import os

# Load .env file from project root if it exists
env_path = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    # Strip spaces and optional quotes
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")
    except Exception:
        pass

# API Key handling
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

# Default model selection - prioritizing environment first, defaulting to gemini-2.5-flash
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Model generation parameters
DEFAULT_MAX_TOKENS = int(os.environ.get("GEMINI_MAX_TOKENS", "2048"))
DEFAULT_TIMEOUT_SECS = float(os.environ.get("GEMINI_TIMEOUT", "30.0"))

def get_client():
    """
    Retrieves a google-genai Client initialized with environmental API keys.
    
    Raises:
        ValueError: If no API key is set in environment variables.
    """
    from google import genai
    
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "API Key missing! Please set the 'GEMINI_API_KEY' or 'GOOGLE_API_KEY' "
            "environment variable to enable natural language capabilities."
        )
    return genai.Client(api_key=api_key)

def is_api_configured() -> bool:
    """Checks if the environment is configured with a valid API key."""
    return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
