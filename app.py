from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import random
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# API Keys from environment variables
GEMINI_API_KEY = os.environ.get('Gemini_API_key', 'AIzaSyD8Vb3TXMsoWVC9FAzBmdOXdhTHogBZeXk')
WEATHER_API_KEY = os.environ.get('Weather_API_key')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-pro')

logger.info(f"Gemini API configured: {bool(GEMINI_API_KEY)}")
logger.info(f"Weather API configured: {bool(WEATHER_API_KEY)}")

# Try to import optional dependencies
try:
    import google.generativeai as genai
    genai.configure(api_key=GEMINI_API_KEY)
    GENAI_AVAILABLE = True
    logger.info("✅ Google GenerativeAI imported and configured")
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("❌ Google GenerativeAI not available")

try:
    import requests
    REQUESTS_AVAILABLE = True
    logger.info("✅ Requests library available")
except ImportError:
    REQUESTS_AVAILABLE = False
    logger.warning("❌ Requests library not available")

# Agricultural Knowledge Base for Fallback
agricultural_knowledge = {
    'rice': {
        'keywords': ['rice', 'paddy', 'chawal', 'धान'],
        'responses': [
            "🌾 Rice grows best in flooded fields with temperatures 20-35°C. Plant during monsoon (June-July) for Kharif season.",
            "🌾 For rice cultivation: Use 120kg N, 60kg P2O5, 40kg K2O per hectare. Maintain 2-5cm water level.",
            "🌾 Rice varieties: Basmati for export, IR64 for high yield. Harvest when 80% grains turn golden yellow."
        ]
    },
    'wheat': {
        'keywords': ['wheat', 'gehun', 'गेहूं'],
        'responses': [
            "🌾 Wheat is a Rabi crop. Sow in November-December, harvest in March-April. Needs 15-25°C temperature.",
            "🌾 For wheat: Apply 150kg N, 75kg P2O5, 60kg K2O per hectare. Irrigate 4-6 times during growing season.",
            "🌾 Popular wheat varieties: HD2967, PBW343, DBW17. Ensure proper drainage to prevent waterlogging."
        ]
    },
    'fertilizer': {
        'keywords': ['fertilizer', 'khad', 'खाद', 'urea', 'dap', 'npk'],
        'responses': [
            "🌱 NPK fertilizers: N for leaf growth, P for roots/flowers, K for disease resistance. Test soil before applying.",
            "🌱 Organic fertilizers: Compost, vermicompost, green manure improve soil health long-term.",
            "🌱 Apply fertilizers in split doses: 1/3 at sowing, 1/3 at vegetative stage, 1/3 at flowering."
        ]
    }
}

def get_fallback_response(user_message):
    """Generate intelligent fallback response based on agricultural knowledge"""
    user_message_lower = user_message.lower()
    
    # Check for specific crop mentions
    for crop, data in agricultural_knowledge.items():
        if 'keywords' in data:
            for keyword in data['keywords']:
                if keyword.lower() in user_message_lower:
                    return random.choice(data['responses'])
    
    # Default helpful response
    return "🌾 I'm here to help with your farming questions! You can ask me about:\n• Crop cultivation (rice, wheat, maize, cotton, etc.)\n• Fertilizers and soil management\n• Irrigation and pest control\n• Seeds and varieties\n• Organic farming\n\nPlease ask a specific question about any farming topic!"

@app.route('/')
def home():
    return jsonify({
        'message': 'KrishiMitra API Running', 
        'status': 'OK',
        'version': 'full',
        'gemini_configured': bool(GEMINI_API_KEY),
        'weather_configured': bool(WEATHER_API_KEY),
        'services': {
            'genai': GENAI_AVAILABLE,
            'requests': REQUESTS_AVAILABLE
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'services': {
            'gemini_ai': GENAI_AVAILABLE,
            'weather_api': REQUESTS_AVAILABLE,
            'knowledge_base': True
        },
        'timestamp': datetime.datetime.now().isoformat()
    })

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_msg = data.get('message', '')
    lang = data.get('lang', 'en-US')
    concise = bool(data.get('concise', True))
    
    if not user_msg:
        return jsonify({'success': False, 'error': 'No message provided'}), 400
    
    logger.info(f"Chatbot request: {user_msg}")
    
    # Try Gemini first if available
    if GENAI_AVAILABLE and GEMINI_API_KEY:
        gemini_models = ['gemini-pro', 'gemini-1.5-flash', 'models/gemini-pro']
        
        for model_name in gemini_models:
            try:
                logger.info(f"Trying Gemini model: {model_name}")
                model_ai = genai.GenerativeModel(model_name)
                style = 'Answer very concisely in 1-3 sentences.' if concise else 'Answer clearly and helpfully.'
                prompt = f"You are a farming expert. {style} Question: {user_msg}"
                resp = model_ai.generate_content(prompt)
                text = (resp.text or '').strip()
                
                if text and len(text) > 10:
                    logger.info(f"Successfully used Gemini model: {model_name}")
                    return jsonify({
                        'success': True, 
                        'response': text, 
                        'lang': lang, 
                        'concise': concise,
                        'source': 'gemini',
                        'model_used': model_name
                    })
            except Exception as gemini_error:
                logger.warning(f"Gemini model {model_name} failed: {gemini_error}")
                continue
    
    # Fallback to agricultural knowledge base
    try:
        fallback_response = get_fallback_response(user_msg)
        logger.info(f"Using knowledge base fallback")
        
        return jsonify({
            'success': True, 
            'response': fallback_response, 
            'lang': lang, 
            'concise': concise,
            'source': 'knowledge_base'
        })
        
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({
            'success': True, 
            'response': '🌾 I\'m here to help with farming questions! You can ask me about crops like rice, wheat, maize, cotton, fertilizers, irrigation, pest control, and more.',
            'source': 'emergency_fallback'
        })

@app.route('/api/weather', methods=['POST'])
def weather():
    if not REQUESTS_AVAILABLE:
        return jsonify({'success': False, 'error': 'Weather service not available'}), 503
        
    data = request.json
    lat = data.get('latitude', 19.076)
    lon = data.get('longitude', 72.8777)
    
    # Mock weather response for now (can be enhanced later)
    return jsonify({
        'success': True,
        'location': {'city': 'Mumbai', 'country': 'IN'},
        'current': {
            'temperature': 28,
            'humidity': 75,
            'condition': 'partly cloudy',
            'windSpeed': 15,
            'precipitation': 0
        },
        'forecast': [
            {'date': '2025-09-27', 'maxTemp': 30, 'minTemp': 24, 'condition': 'sunny'},
            {'date': '2025-09-28', 'maxTemp': 29, 'minTemp': 23, 'condition': 'cloudy'},
            {'date': '2025-09-29', 'maxTemp': 27, 'minTemp': 22, 'condition': 'rainy'}
        ],
        'agricultural_advisory': [
            {'title': 'Good Weather for Farming', 'description': 'Ideal conditions for field operations and crop growth.'}
        ]
    })

@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    current_month = datetime.datetime.now().month
    
    return jsonify({
        'success': True,
        'stats': {
            'total_predictions': {'value': '12,500+', 'growth': '+15%'},
            'farmers_helped': {'value': '3,240', 'growth': '+8%'},
            'crop_varieties': {'value': '12', 'growth': '+20%'},
            'success_rate': {'value': '94.2%', 'growth': '+3%'}
        },
        'last_updated': datetime.datetime.now().isoformat()
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting KrishiMitra API on port {port}")
    logger.info(f"Gemini API configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"Weather API configured: {bool(WEATHER_API_KEY)}")
    app.run(host='0.0.0.0', port=port, debug=False)
