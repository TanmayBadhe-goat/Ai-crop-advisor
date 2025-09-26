#!/usr/bin/env python3
import os
import sys
import logging

# Set up logging first
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("=== Starting KrishiMitra Simple Test ===")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"PORT environment variable: {os.environ.get('PORT', 'Not set')}")

try:
    from flask import Flask, jsonify
    logger.info("✅ Flask imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Flask: {e}")
    sys.exit(1)

try:
    from flask_cors import CORS
    logger.info("✅ Flask-CORS imported successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import Flask-CORS: {e}")
    sys.exit(1)

# Create Flask app
app = Flask(__name__)
CORS(app)

logger.info("✅ Flask app created successfully")

@app.route('/', methods=['GET'])
def home():
    logger.info("Home endpoint accessed")
    return jsonify({
        'message': 'KrishiMitra Simple Test - Working!', 
        'status': 'OK',
        'python_version': sys.version,
        'port': os.environ.get('PORT', 'Not set'),
        'cwd': os.getcwd()
    })

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health endpoint accessed")
    return jsonify({'status': 'healthy'})

@app.route('/test', methods=['GET'])
def test():
    logger.info("Test endpoint accessed")
    return jsonify({'message': 'Test successful'})

# Error handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}")
    return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        port = int(os.environ.get('PORT', 5000))
        logger.info(f"Starting Flask app on 0.0.0.0:{port}")
        
        # Start the app
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start Flask app: {e}")
        sys.exit(1)
