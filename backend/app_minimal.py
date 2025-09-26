from flask import Flask, jsonify
from flask_cors import CORS
import os
import logging
import datetime

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting minimal KrishiMitra API...")

@app.route('/', methods=['GET'])
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        'message': 'KrishiMitra API Running (Minimal Version)', 
        'status': 'OK',
        'timestamp': datetime.datetime.now().isoformat(),
        'port': os.environ.get('PORT', 'Not set')
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check"""
    logger.info("Health check endpoint accessed")
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint"""
    return jsonify({
        'message': 'Test endpoint working',
        'environment_vars': {
            'PORT': os.environ.get('PORT'),
            'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT'),
            'RAILWAY_SERVICE_NAME': os.environ.get('RAILWAY_SERVICE_NAME')
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting minimal app on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)
