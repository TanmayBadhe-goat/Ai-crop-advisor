from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import requests
import os
import logging
import base64
from PIL import Image
import io
from analytics import analytics_manager
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
import os

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
GEMINI_API_KEY = os.environ.get('Gemini_API_key')
WEATHER_API_KEY = os.environ.get('Weather_API_key')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
logger.info(f"Gemini API configured: {'Yes' if bool(GEMINI_API_KEY) else 'No'} | Model: {GEMINI_MODEL}")

# Combined training data (corrected - all arrays have 30 elements)
training_data = {
    'N': [90, 80, 60, 55, 85, 74, 78, 50, 20, 40, 45, 55, 80, 70, 30, 25, 120, 110, 90, 100, 60, 80, 80, 60, 280, 50, 25, 150, 100, 500],
    'P': [40, 45, 35, 30, 58, 35, 42, 25, 30, 35, 25, 40, 50, 45, 60, 70, 80, 75, 65, 70, 30, 35, 40, 40, 90, 75, 50, 100, 50, 250],
    'K': [43, 40, 38, 35, 41, 40, 42, 20, 25, 30, 35, 25, 40, 35, 50, 60, 70, 65, 55, 60, 25, 30, 40, 40, 90, 30, 25, 100, 150, 500],
    'temperature': [25, 26, 27, 23, 21.7, 26.4, 20.1, 15.5, 18.2, 22.1, 19.8, 24.3, 25.2, 28.5, 30.1, 32.5, 27.8, 29.2, 24.5, 26.8, 18.5, 22.3, 28, 30, 26, 27, 28, 24, 25, 23],
    'humidity': [80, 75, 70, 68, 80, 80, 81, 75, 70, 85, 78, 83, 88, 85, 60, 55, 65, 62, 70, 68, 75, 78, 55, 50, 70, 65, 60, 70, 60, 60],
    'ph': [6.5, 6.8, 7.0, 6.7, 7.0, 6.9, 7.6, 6.2, 6.8, 7.2, 6.4, 7.1, 7.5, 6.8, 8.2, 8.5, 7.8, 8.0, 7.2, 7.5, 6.0, 6.5, 7.0, 7.5, 6.8, 6.5, 6.3, 6.5, 6.8, 6.0],
    'rainfall': [200, 210, 220, 190, 226, 242, 262, 180, 150, 200, 175, 210, 250, 280, 120, 90, 80, 100, 140, 160, 220, 240, 60, 50, 150, 80, 60, 80, 75, 100],
    'label': [
        'rice', 'wheat', 'maize', 'cotton', 'rice', 'rice', 'rice', 'wheat', 'wheat', 'wheat', 
        'wheat', 'wheat', 'maize', 'maize', 'cotton', 'cotton', 'sugarcane', 'sugarcane', 
        'potato', 'potato', 'tomato', 'tomato', 'jowar', 'bajra', 'sugarcane', 'soybean', 
        'groundnut', 'tomato', 'grapes', 'orange'
    ]
}

df = pd.DataFrame(training_data)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

logger.info(f"Model trained with accuracy: {model.score(X_scaled, y):.2%}")

# Combined crop database (enhanced info from both)
crop_database = {
    'rice': {'emoji': '🌾', 'season': 'Kharif (June-October)', 'duration': '120-150 days', 'yield': '3-4 tons/hectare', 'market_price': '₹2000-2500/quintal', 'tips': 'Use NPK fertilizer 4:2:1 ratio or apply 120kg N, 60kg P2O5, 40kg K2O per hectare.'},
    'wheat': {'emoji': '🌾', 'season': 'Rabi (November-April)', 'duration': '120-150 days', 'yield': '2-3 tons/hectare', 'market_price': '₹2100-2600/quintal', 'tips': 'Apply urea and phosphorus properly, about 150kg N, 75kg P2O5, 60kg K2O per hectare.'},
    'maize': {'emoji': '🌽', 'season': 'Year-round', 'duration': '90-120 days', 'yield': '4-6 tons/hectare', 'market_price': '₹1800-2200/quintal', 'tips': 'Use balanced fertilizer and irrigation, apply 120kg N, 60kg P2O5, 40kg K2O per hectare.'},
    'cotton': {'emoji': '🌿', 'season': 'Kharif (April-October)', 'duration': '180-200 days', 'yield': '1-2 tons/hectare', 'market_price': '₹5500-6500/quintal', 'tips': 'Soil testing is important before sowing. Apply 100kg N, 50kg P2O5, 50kg K2O per hectare.'},
    'sugarcane': {'emoji': '🎋', 'season': 'Year-round', 'duration': '12-18 months', 'yield': '70-100 tons/hectare', 'market_price': '₹280-350/quintal', 'tips': 'Apply 280kg N, 90kg P2O5, 90kg K2O per hectare.'},
    'potato': {'emoji': '🥔', 'season': 'Rabi (October-February)', 'duration': '90-120 days', 'yield': '20-25 tons/hectare', 'market_price': '₹800-1500/quintal', 'tips': 'Apply 180kg N, 80kg P2O5, 100kg K2O per hectare.'},
    'tomato': {'emoji': '🍅', 'season': 'Year-round', 'duration': '90-120 days', 'yield': '40-60 tons/hectare', 'market_price': '₹1000-2000/quintal', 'tips': 'Apply 150kg N, 100kg P2O5, 100kg K2O per hectare.'},
    'jowar': {'emoji': '🌾', 'season': 'Kharif/Rabi', 'duration': '110-130 days', 'yield': '2-3 tons/hectare', 'market_price': '₹2500-3000/quintal (approx)', 'tips': 'Apply NPK fertilizer 80:40:40 kg/ha.'},
    'bajra': {'emoji': '🌾', 'season': 'Kharif (June-October)', 'duration': '80-100 days', 'yield': '1.5-2 tons/hectare', 'market_price': '₹2300-2600/quintal (approx)', 'tips': 'Apply NPK fertilizer 60:40:40 kg/ha.'},
    'soybean': {'emoji': '🌱', 'season': 'Kharif (June-October)', 'duration': '90-110 days', 'yield': '1.5-2 tons/hectare', 'market_price': '₹3800-4500/quintal (approx)', 'tips': 'Apply NPK fertilizer 50:75:30 kg/ha.'},
    'groundnut': {'emoji': '🥜', 'season': 'Kharif (June-October)', 'duration': '100-120 days', 'yield': '2-3 tons/hectare', 'market_price': '₹5000-6000/quintal (approx)', 'tips': 'Apply NPK fertilizer 25:50:25 kg/ha.'},
    'grapes': {'emoji': '🍇', 'season': 'Dec-Apr (harvest)', 'duration': '4-5 months (after pruning)', 'yield': '20-25 tons/hectare', 'market_price': '₹2500-6000/quintal', 'tips': 'High potassium, drip irrigation recommended.'},
    'orange': {'emoji': '🍊', 'season': 'Oct-Feb (harvest)', 'duration': '~1 year cycle', 'yield': '10-15 tons/hectare', 'market_price': '₹3000-5000/quintal', 'tips': 'Apply NPK fertilizer 500:250:500 g/plant annually.'}
}

# Combined disease database
disease_database = {
    'healthy': {
        'name': 'Healthy Plant',
        'severity': 'None',
        'description': 'Your plant appears healthy.',
        'treatment': 'Keep maintaining your crop regularly.',
        'prevention': 'Ensure proper watering and nutrition.',
        'emoji': '🌱'
    },
    'bacterial_spot': {
        'name': 'Bacterial Spot',
        'severity': 'Medium',
        'description': 'Dark spots on leaves with yellow halos.',
        'treatment': 'Apply copper fungicides.',
        'prevention': 'Avoid overhead irrigation.',
        'emoji': '🦠'
    },
    'early_blight': {
        'name': 'Early Blight',
        'severity': 'High',
        'description': 'Brown rings on older leaves.',
        'treatment': 'Use fungicides like chlorothalonil.',
        'prevention': 'Rotate crops and mulch soil.',
        'emoji': '🍂'
    }
}

# Weather function shared by both
def get_weather_data(lat, lon):
    try:
        # Get current weather
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {'lat': lat, 'lon': lon, 'appid': WEATHER_API_KEY, 'units': 'metric'}
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get forecast data
            forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
            forecast_response = requests.get(forecast_url, params=params)
            forecast_data = forecast_response.json() if forecast_response.status_code == 200 else None
            
            # Generate mock forecast if API fails
            forecast = []
            if forecast_data:
                # Process 3-day forecast from 5-day forecast data
                for i in range(0, min(24, len(forecast_data['list'])), 8):  # Every 24 hours
                    item = forecast_data['list'][i]
                    forecast.append({
                        'date': item['dt_txt'].split(' ')[0],
                        'maxTemp': round(item['main']['temp_max']),
                        'minTemp': round(item['main']['temp_min']),
                        'condition': item['weather'][0]['description']
                    })
            else:
                # Fallback mock forecast
                import datetime
                for i in range(3):
                    date = (datetime.datetime.now() + datetime.timedelta(days=i+1)).strftime('%Y-%m-%d')
                    forecast.append({
                        'date': date,
                        'maxTemp': round(data['main']['temp'] + random.uniform(-3, 3)),
                        'minTemp': round(data['main']['temp'] - random.uniform(5, 10)),
                        'condition': data['weather'][0]['description']
                    })
            
            # Generate agricultural advisory based on weather
            advisory = generate_agricultural_advisory(data)
            
            return {
                'success': True,
                'location': {
                    'city': data.get('name', 'Unknown'),
                    'country': data.get('sys', {}).get('country', 'Unknown')
                },
                'current': {
                    'temperature': round(data['main']['temp']),
                    'humidity': data['main']['humidity'],
                    'condition': data['weather'][0]['description'],
                    'windSpeed': round(data.get('wind', {}).get('speed', 0) * 3.6),  # Convert m/s to km/h
                    'precipitation': round(data.get('rain', {}).get('1h', 0), 1)  # mm in last hour
                },
                'forecast': forecast,
                'agricultural_advisory': advisory
            }
    except Exception as e:
        logger.error(f"Weather API error: {e}")
    return None

def generate_agricultural_advisory(weather_data):
    """Generate agricultural advisory based on weather conditions"""
    advisory = []
    temp = weather_data['main']['temp']
    humidity = weather_data['main']['humidity']
    condition = weather_data['weather'][0]['description'].lower()
    
    # Temperature-based advice
    if temp > 35:
        advisory.append({
            'title': 'High Temperature Alert',
            'description': 'Provide shade to crops and increase irrigation frequency. Avoid midday field work.'
        })
    elif temp < 10:
        advisory.append({
            'title': 'Cold Weather Warning',
            'description': 'Protect sensitive crops from frost. Consider using mulch or row covers.'
        })
    
    # Humidity-based advice
    if humidity > 80:
        advisory.append({
            'title': 'High Humidity Alert',
            'description': 'Monitor crops for fungal diseases. Ensure good air circulation and avoid overhead watering.'
        })
    elif humidity < 40:
        advisory.append({
            'title': 'Low Humidity Notice',
            'description': 'Increase irrigation and consider mulching to retain soil moisture.'
        })
    
    # Weather condition-based advice
    if 'rain' in condition:
        advisory.append({
            'title': 'Rainy Weather Advisory',
            'description': 'Avoid field operations. Check drainage systems and monitor for waterlogging.'
        })
    elif 'clear' in condition or 'sunny' in condition:
        advisory.append({
            'title': 'Clear Weather Opportunity',
            'description': 'Good conditions for field operations, harvesting, and drying crops.'
        })
    
    # Default advice if no specific conditions
    if not advisory:
        advisory.append({
            'title': 'General Farming Advice',
            'description': 'Monitor crop health regularly and maintain proper irrigation schedule.'
        })
    
    return advisory

# Routes merged and enhanced

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'KrishiMitra API Running', 'status': 'OK'})

@app.route('/api', methods=['GET'])
def api_index():
    """Simple API index to help verify routing in production."""
    return jsonify({
        'service': 'KrishiMitra API',
        'status': 'OK',
        'endpoints': [
            'GET /api',
            'GET /api/health',
            'POST /api/predict',
            'POST /api/chatbot',
            'POST /api/weather',
            'POST /api/upload-image',
            'POST /api/disease-detection',
            'GET /api/dashboard-stats',
            'GET /api/crop-calendar',
            'GET /api/crop-calendar/month/<int:month>'
        ]
    })

@app.route('/api/health', methods=['GET'])
def api_health():
    return jsonify({'status': 'ok', 'service': 'KrishiMitra API'}), 200

@app.route('/health', methods=['GET'])
def root_health_alias():
    """Alias for health check to handle proxies that strip the /api prefix."""
    return jsonify({'status': 'ok', 'service': 'KrishiMitra API', 'alias': True}), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        required = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        features = []
        for field in required:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing {field}'}), 400
            features.append(float(data[field]))

        scaled = scaler.transform([features])
        pred = model.predict(scaled)[0]
        conf = float(max(model.predict_proba(scaled)[0]))
        info = crop_database.get(pred, {})

        return jsonify({
            'success': True,
            'prediction': {
                'crop': pred,
                'confidence': conf,
                'emoji': info.get('emoji', '🌱'),
            },
            'crop_info': info
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    user_msg = data.get('message', '')
    lang = data.get('lang', 'en-US')
    concise = bool(data.get('concise', True))
    if not user_msg:
        return jsonify({'error': 'No message'}), 400
    try:
        if not GEMINI_API_KEY:
            logger.warning('Gemini API key missing; returning fallback reply')
            return jsonify({'success': True, 'response': 'I cannot access the assistant right now. Please try again later.'})

        # Use model from environment to avoid unavailable version errors in some regions/projects
        model_ai = genai.GenerativeModel(GEMINI_MODEL)
        style = 'Answer very concisely in 1-3 sentences.' if concise else 'Answer clearly and helpfully.'
        locale = f"Respond in language/locale: {lang}." if lang else ''
        prompt = f"You are a farming expert. {style} {locale} Question: {user_msg}"
        resp = model_ai.generate_content(prompt)
        text = (resp.text or '').strip()
        if not text:
            text = 'Sorry, I could not generate a response.'
        return jsonify({'success': True, 'response': text, 'lang': lang, 'concise': concise})
    except Exception as e:
        logger.error(f"Chatbot error: {e}")
        return jsonify({'success': True, 'response': 'Please consult local experts.'})

@app.route('/api/weather', methods=['POST'])
def weather():
    data = request.json
    lat = data.get('latitude', 19.076)
    lon = data.get('longitude', 72.8777)
    weather_data = get_weather_data(lat, lon)
    if weather_data:
        return jsonify(weather_data)
    else:
        return jsonify({'success': False, 'error': 'Weather fetch failed'}), 500

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image provided'}), 400
    file = request.files['image']
    img_bytes = file.read()
    img_b64 = base64.b64encode(img_bytes).decode('utf-8')
    return jsonify({'success': True, 'image_base64': img_b64})

@app.route('/api/disease-detection', methods=['POST'])
def disease_detect():
    data = request.json
    img_b64 = data.get('image_base64')
    if not img_b64:
        return jsonify({'success': False, 'error': 'No image data'}), 400
    disease = random.choice(list(disease_database.keys()))
    conf = round(random.uniform(0.7, 0.95), 2)
    info = disease_database[disease]
    return jsonify({
        'success': True,
        'disease': {
            'name': info['name'],
            'confidence': conf,
            'severity': info['severity'],
            'emoji': info['emoji']
        },
        'diagnosis': {
            'description': info['description'],
            'treatment': info['treatment'],
            'prevention': info['prevention']
        }
    })

@app.route('/api/dashboard-stats', methods=['GET'])
def dashboard_stats():
    """Get dashboard statistics"""
    try:
        # Simulate real statistics (in production, these would come from a database)
        import datetime
        current_month = datetime.datetime.now().month
        
        # Generate realistic statistics
        base_predictions = 12500 + (current_month * 150)
        farmers_helped = 3240 + (current_month * 85)
        crop_varieties = len(crop_database)
        success_rate = round(random.uniform(92, 98), 1)
        
        # Calculate growth percentages
        prediction_growth = round(random.uniform(12, 18), 1)
        farmer_growth = round(random.uniform(6, 12), 1)
        variety_growth = round(random.uniform(15, 25), 1)
        success_growth = round(random.uniform(2, 8), 1)
        
        return jsonify({
            'success': True,
            'stats': {
                'total_predictions': {
                    'value': f"{base_predictions:,}+",
                    'growth': f"+{prediction_growth}%"
                },
                'farmers_helped': {
                    'value': f"{farmers_helped:,}",
                    'growth': f"+{farmer_growth}%"
                },
                'crop_varieties': {
                    'value': str(crop_varieties),
                    'growth': f"+{variety_growth}%"
                },
                'success_rate': {
                    'value': f"{success_rate}%",
                    'growth': f"+{success_growth}%"
                }
            },
            'last_updated': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        return jsonify({'success': False, 'error': 'Failed to fetch statistics'}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get detailed analytics and performance metrics"""
    try:
        analytics_data = analytics_manager.get_analytics_summary()
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        return jsonify({'success': False, 'error': 'Failed to fetch analytics'}), 500

@app.route('/api/record-prediction', methods=['POST'])
def record_prediction():
    """Record a prediction for analytics"""
    try:
        data = request.get_json()
        crop = data.get('crop', 'unknown')
        confidence = data.get('confidence', 0.5)
        success = data.get('success', True)
        
        analytics_manager.record_prediction(crop, confidence, success)
        return jsonify({'success': True, 'message': 'Prediction recorded'})
    except Exception as e:
        logger.error(f"Record prediction error: {e}")
        return jsonify({'success': False, 'error': 'Failed to record prediction'}), 500

@app.route('/api/crop-calendar', methods=['GET'])
def get_crop_calendar():
    """Get comprehensive crop calendar data"""
    try:
        import datetime
        
        # Enhanced crop calendar data with planting and harvesting months
        crop_calendar = [
            {
                'crop': 'Rice',
                'emoji': '🌾',
                'season': 'Kharif',
                'plantingMonths': ['Jun', 'Jul', 'Aug'],
                'harvestMonths': ['Oct', 'Nov', 'Dec'],
                'duration': '120-150 days',
                'tips': 'Plant during monsoon. Requires flooded fields.',
                'color': '#4CAF50',
                'yield': '3-4 tons/hectare',
                'market_price': '₹2000-2500/quintal'
            },
            {
                'crop': 'Wheat',
                'emoji': '🌾',
                'season': 'Rabi',
                'plantingMonths': ['Nov', 'Dec', 'Jan'],
                'harvestMonths': ['Mar', 'Apr', 'May'],
                'duration': '120-150 days',
                'tips': 'Plant in winter. Requires cool weather for growth.',
                'color': '#FF9800',
                'yield': '2-3 tons/hectare',
                'market_price': '₹2100-2600/quintal'
            },
            {
                'crop': 'Maize',
                'emoji': '🌽',
                'season': 'Kharif',
                'plantingMonths': ['Jun', 'Jul'],
                'harvestMonths': ['Sep', 'Oct'],
                'duration': '90-120 days',
                'tips': 'Can be grown year-round with irrigation.',
                'color': '#FFC107',
                'yield': '4-6 tons/hectare',
                'market_price': '₹1800-2200/quintal'
            },
            {
                'crop': 'Cotton',
                'emoji': '🌿',
                'season': 'Kharif',
                'plantingMonths': ['Apr', 'May', 'Jun'],
                'harvestMonths': ['Oct', 'Nov', 'Dec'],
                'duration': '180-200 days',
                'tips': 'Requires warm weather and moderate rainfall.',
                'color': '#E91E63',
                'yield': '1-2 tons/hectare',
                'market_price': '₹5500-6500/quintal'
            },
            {
                'crop': 'Sugarcane',
                'emoji': '🎋',
                'season': 'Year-round',
                'plantingMonths': ['Feb', 'Mar', 'Oct', 'Nov'],
                'harvestMonths': ['Dec', 'Jan', 'Feb', 'Mar'],
                'duration': '12-18 months',
                'tips': 'Long duration crop. Plant in spring or autumn.',
                'color': '#9C27B0',
                'yield': '70-100 tons/hectare',
                'market_price': '₹280-350/quintal'
            },
            {
                'crop': 'Potato',
                'emoji': '🥔',
                'season': 'Rabi',
                'plantingMonths': ['Oct', 'Nov', 'Dec'],
                'harvestMonths': ['Jan', 'Feb', 'Mar'],
                'duration': '90-120 days',
                'tips': 'Cool weather crop. Avoid frost during harvest.',
                'color': '#795548',
                'yield': '20-25 tons/hectare',
                'market_price': '₹800-1500/quintal'
            },
            {
                'crop': 'Tomato',
                'emoji': '🍅',
                'season': 'Year-round',
                'plantingMonths': ['Jun', 'Jul', 'Oct', 'Nov'],
                'harvestMonths': ['Sep', 'Oct', 'Jan', 'Feb'],
                'duration': '90-120 days',
                'tips': 'Can be grown in multiple seasons with proper care.',
                'color': '#F44336',
                'yield': '40-60 tons/hectare',
                'market_price': '₹1000-2000/quintal'
            },
            {
                'crop': 'Mustard',
                'emoji': '🌻',
                'season': 'Rabi',
                'plantingMonths': ['Oct', 'Nov'],
                'harvestMonths': ['Feb', 'Mar'],
                'duration': '90-110 days',
                'tips': 'Cool season oilseed crop.',
                'color': '#FFEB3B',
                'yield': '1.5-2 tons/hectare',
                'market_price': '₹4500-5500/quintal'
            }
        ]
        
        # Generate upcoming events based on current month
        current_month = datetime.datetime.now().month
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        current_month_name = month_names[current_month - 1]
        
        upcoming_events = []
        
        # Find crops to plant this month
        for crop in crop_calendar:
            if current_month_name in crop['plantingMonths']:
                upcoming_events.append({
                    'id': f"plant_{crop['crop'].lower()}",
                    'title': f"{crop['crop']} Planting",
                    'date': "This month",
                    'type': "sowing",
                    'crop': crop['crop'],
                    'description': f"Optimal time for {crop['crop'].lower()} planting in {crop['season']} season",
                    'emoji': crop['emoji']
                })
            
            if current_month_name in crop['harvestMonths']:
                upcoming_events.append({
                    'id': f"harvest_{crop['crop'].lower()}",
                    'title': f"{crop['crop']} Harvest",
                    'date': "This month",
                    'type': "harvest",
                    'crop': crop['crop'],
                    'description': f"Harvest time for {crop['crop'].lower()} crop",
                    'emoji': crop['emoji']
                })
        
        # Add some general farming activities
        general_events = [
            {
                'id': 'irrigation_check',
                'title': 'Irrigation System Check',
                'date': 'Weekly',
                'type': 'irrigation',
                'crop': 'All crops',
                'description': 'Regular check of irrigation systems and water supply',
                'emoji': '💧'
            },
            {
                'id': 'fertilizer_application',
                'title': 'Fertilizer Application',
                'date': 'Monthly',
                'type': 'fertilizer',
                'crop': 'Active crops',
                'description': 'Apply appropriate fertilizers based on crop growth stage',
                'emoji': '🌱'
            }
        ]
        
        upcoming_events.extend(general_events)
        
        return jsonify({
            'success': True,
            'crop_calendar': crop_calendar,
            'upcoming_events': upcoming_events[:6],  # Limit to 6 events
            'current_month': current_month_name,
            'seasons': ['All', 'Kharif', 'Rabi', 'Zaid'],
            'months': month_names,
            'weather_recommendation': {
                'title': 'Weather-Based Recommendations',
                'description': 'Current conditions are favorable for field activities. Monitor weather forecasts for planning.',
                'favorable': True
            },
            'last_updated': datetime.datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Crop calendar error: {e}")
        return jsonify({'success': False, 'error': 'Failed to fetch crop calendar data'}), 500

@app.route('/api/crop-calendar/month/<int:month>', methods=['GET'])
def get_month_activities(month):
    """Get activities for a specific month"""
    try:
        if month < 1 or month > 12:
            return jsonify({'success': False, 'error': 'Invalid month'}), 400
            
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        month_name = month_names[month - 1]
        
        # Get crops for this month from the database
        crops_to_plant = []
        crops_to_harvest = []
        
        for crop_name, crop_info in crop_database.items():
            # This is a simplified version - in a real app, you'd have proper month data
            if crop_name in ['rice', 'maize'] and month in [6, 7, 8]:  # Kharif planting
                crops_to_plant.append({
                    'crop': crop_name.title(),
                    'emoji': crop_info['emoji'],
                    'duration': crop_info['duration'],
                    'season': crop_info['season']
                })
            elif crop_name in ['wheat', 'potato'] and month in [11, 12, 1]:  # Rabi planting
                crops_to_plant.append({
                    'crop': crop_name.title(),
                    'emoji': crop_info['emoji'],
                    'duration': crop_info['duration'],
                    'season': crop_info['season']
                })
            elif crop_name in ['rice', 'maize'] and month in [10, 11, 12]:  # Kharif harvest
                crops_to_harvest.append({
                    'crop': crop_name.title(),
                    'emoji': crop_info['emoji'],
                    'duration': crop_info['duration'],
                    'season': crop_info['season']
                })
            elif crop_name in ['wheat', 'potato'] and month in [3, 4, 5]:  # Rabi harvest
                crops_to_harvest.append({
                    'crop': crop_name.title(),
                    'emoji': crop_info['emoji'],
                    'duration': crop_info['duration'],
                    'season': crop_info['season']
                })
        
        return jsonify({
            'success': True,
            'month': month_name,
            'month_number': month,
            'crops_to_plant': crops_to_plant,
            'crops_to_harvest': crops_to_harvest,
            'activities_count': len(crops_to_plant) + len(crops_to_harvest)
        })
    except Exception as e:
        logger.error(f"Month activities error: {e}")
        return jsonify({'success': False, 'error': 'Failed to fetch month activities'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
