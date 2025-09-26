from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import datetime

app = Flask(__name__)
CORS(app)

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting KrishiMitra API with debug mode...")

# Try to import optional dependencies with error handling
try:
    import google.generativeai as genai
    logger.info("✅ google.generativeai imported successfully")
    GENAI_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import google.generativeai: {e}")
    GENAI_AVAILABLE = False

try:
    import requests
    logger.info("✅ requests imported successfully")
    REQUESTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import requests: {e}")
    REQUESTS_AVAILABLE = False

try:
    from PIL import Image
    logger.info("✅ PIL imported successfully")
    PIL_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import PIL: {e}")
    PIL_AVAILABLE = False

try:
    from analytics import analytics_manager
    logger.info("✅ analytics imported successfully")
    ANALYTICS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import analytics: {e}")
    ANALYTICS_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    import numpy as np
    logger.info("✅ ML libraries imported successfully")
    ML_LIBS_AVAILABLE = True
except ImportError as e:
    logger.error(f"❌ Failed to import ML libraries: {e}")
    ML_LIBS_AVAILABLE = False

# API Keys
GEMINI_API_KEY = os.environ.get('Gemini_API_key', 'AIzaSyD8Vb3TXMsoWVC9FAzBmdOXdhTHogBZeXk')
WEATHER_API_KEY = os.environ.get('Weather_API_key')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-pro')

logger.info(f"Gemini API Key configured: {bool(GEMINI_API_KEY)}")
logger.info(f"Weather API Key configured: {bool(WEATHER_API_KEY)}")

@app.route('/', methods=['GET'])
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        'message': 'KrishiMitra API Running (Debug Version)', 
        'status': 'OK',
        'dependencies': {
            'genai': GENAI_AVAILABLE,
            'requests': REQUESTS_AVAILABLE,
            'pil': PIL_AVAILABLE,
            'analytics': ANALYTICS_AVAILABLE,
            'ml_libs': ML_LIBS_AVAILABLE
        },
        'config': {
            'gemini_configured': bool(GEMINI_API_KEY),
            'weather_configured': bool(WEATHER_API_KEY),
            'gemini_model': GEMINI_MODEL
        },
        'environment': {
            'port': os.environ.get('PORT'),
            'railway_env': os.environ.get('RAILWAY_ENVIRONMENT'),
            'railway_service': os.environ.get('RAILWAY_SERVICE_NAME')
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Detailed health check"""
    logger.info("Health check endpoint accessed")
    return jsonify({
        'status': 'healthy',
        'services': {
            'flask': True,
            'genai': GENAI_AVAILABLE,
            'requests': REQUESTS_AVAILABLE,
            'analytics': ANALYTICS_AVAILABLE,
            'ml_libs': ML_LIBS_AVAILABLE
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/test', methods=['GET'])
def test_api():
    """Test API endpoint"""
    return jsonify({
        'message': 'API test successful',
        'timestamp': datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting debug app on port {port}")
    logger.info(f"All dependencies status:")
    logger.info(f"  - GenAI: {GENAI_AVAILABLE}")
    logger.info(f"  - Requests: {REQUESTS_AVAILABLE}")
    logger.info(f"  - PIL: {PIL_AVAILABLE}")
    logger.info(f"  - Analytics: {ANALYTICS_AVAILABLE}")
    logger.info(f"  - ML Libraries: {ML_LIBS_AVAILABLE}")
    app.run(debug=False, host='0.0.0.0', port=port)
