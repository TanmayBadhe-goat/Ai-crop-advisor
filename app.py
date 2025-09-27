from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
import random
import datetime
import pickle
import numpy as np
import base64
import io

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

# Load ML model and scaler for crop prediction
try:
    with open('backend/data/model.pkl', 'rb') as f:
        crop_model = pickle.load(f)
    with open('backend/data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    MODEL_AVAILABLE = True
    logger.info("✅ Crop prediction model loaded successfully")
except Exception as e:
    MODEL_AVAILABLE = False
    crop_model = None
    scaler = None
    logger.warning(f"❌ Could not load crop prediction model: {e}")

# Crop information database
CROP_INFO = {
    'rice': {
        'emoji': '🌾',
        'season': 'Kharif (June-October)',
        'duration': '120-150 days',
        'yield': '40-60 quintals/hectare',
        'market_price': '₹2000-2500/quintal',
        'tips': 'Maintain 2-5cm water level. Apply fertilizers in split doses. Harvest when 80% grains turn golden.'
    },
    'wheat': {
        'emoji': '🌾',
        'season': 'Rabi (November-April)',
        'duration': '120-140 days',
        'yield': '35-50 quintals/hectare',
        'market_price': '₹2100-2400/quintal',
        'tips': 'Sow in well-prepared field. Irrigate 4-6 times. Apply nitrogen in 3 split doses.'
    },
    'maize': {
        'emoji': '🌽',
        'season': 'Kharif & Rabi',
        'duration': '90-120 days',
        'yield': '50-80 quintals/hectare',
        'market_price': '₹1800-2200/quintal',
        'tips': 'Plant in rows with proper spacing. Apply balanced fertilizers. Control stem borer and fall armyworm.'
    },
    'cotton': {
        'emoji': '🌿',
        'season': 'Kharif (April-October)',
        'duration': '180-200 days',
        'yield': '15-25 quintals/hectare',
        'market_price': '₹5500-6500/quintal',
        'tips': 'Deep ploughing required. Monitor for bollworm. Pick cotton when bolls are fully opened.'
    },
    'sugarcane': {
        'emoji': '🎋',
        'season': 'Year-round',
        'duration': '12-18 months',
        'yield': '800-1200 quintals/hectare',
        'market_price': '₹300-350/quintal',
        'tips': 'Plant healthy seed cane. Maintain adequate moisture. Harvest at proper maturity for maximum sugar content.'
    },
    'potato': {
        'emoji': '🥔',
        'season': 'Rabi (October-March)',
        'duration': '90-120 days',
        'yield': '250-400 quintals/hectare',
        'market_price': '₹800-1500/quintal',
        'tips': 'Plant certified seed potatoes. Earth up regularly. Control late blight disease.'
    },
    'tomato': {
        'emoji': '🍅',
        'season': 'Kharif & Rabi',
        'duration': '120-150 days',
        'yield': '400-600 quintals/hectare',
        'market_price': '₹1000-2000/quintal',
        'tips': 'Use disease-resistant varieties. Provide support to plants. Regular pruning increases yield.'
    },
    'banana': {
        'emoji': '🍌',
        'season': 'Year-round',
        'duration': '12-15 months',
        'yield': '400-600 quintals/hectare',
        'market_price': '₹1200-1800/quintal',
        'tips': 'Plant tissue culture plants. Maintain adequate drainage. Remove excess suckers regularly.'
    },
    'coconut': {
        'emoji': '🥥',
        'season': 'Year-round',
        'duration': '6-10 years to bear',
        'yield': '80-120 nuts/palm/year',
        'market_price': '₹15-25/nut',
        'tips': 'Plant hybrid varieties. Provide adequate irrigation. Control rhinoceros beetle and red palm weevil.'
    },
    'apple': {
        'emoji': '🍎',
        'season': 'Temperate regions',
        'duration': '3-5 years to bear',
        'yield': '200-400 quintals/hectare',
        'market_price': '₹4000-8000/quintal',
        'tips': 'Requires cold climate. Proper pruning essential. Control apple scab and codling moth.'
    },
    'grapes': {
        'emoji': '🍇',
        'season': 'Year-round (varies by region)',
        'duration': '2-3 years to bear',
        'yield': '200-400 quintals/hectare',
        'market_price': '₹3000-6000/quintal',
        'tips': 'Requires well-drained soil. Proper training and pruning. Control downy mildew and powdery mildew.'
    },
    'orange': {
        'emoji': '🍊',
        'season': 'Year-round',
        'duration': '3-4 years to bear',
        'yield': '300-500 quintals/hectare',
        'market_price': '₹2000-4000/quintal',
        'tips': 'Plant grafted saplings. Maintain soil pH 6.0-7.5. Control citrus canker and fruit fly.'
    }
}

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

@app.route('/api/predict', methods=['POST'])
def predict_crop():
    """Predict the best crop based on soil and environmental parameters"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']
        missing_fields = [field for field in required_fields if field not in data or data[field] is None]
        
        if missing_fields:
            return jsonify({
                'success': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400
        
        # Extract and validate input values
        try:
            nitrogen = float(data['nitrogen'])
            phosphorus = float(data['phosphorus'])
            potassium = float(data['potassium'])
            temperature = float(data['temperature'])
            humidity = float(data['humidity'])
            ph = float(data['ph'])
            rainfall = float(data['rainfall'])
        except (ValueError, TypeError) as e:
            return jsonify({
                'success': False,
                'error': 'All input values must be valid numbers'
            }), 400
        
        # Validate ranges
        if not (0 <= nitrogen <= 300):
            return jsonify({'success': False, 'error': 'Nitrogen should be between 0-300 kg/hectare'}), 400
        if not (0 <= phosphorus <= 150):
            return jsonify({'success': False, 'error': 'Phosphorus should be between 0-150 kg/hectare'}), 400
        if not (0 <= potassium <= 200):
            return jsonify({'success': False, 'error': 'Potassium should be between 0-200 kg/hectare'}), 400
        if not (0 <= temperature <= 50):
            return jsonify({'success': False, 'error': 'Temperature should be between 0-50°C'}), 400
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity should be between 0-100%'}), 400
        if not (3 <= ph <= 10):
            return jsonify({'success': False, 'error': 'pH should be between 3-10'}), 400
        if not (0 <= rainfall <= 3000):
            return jsonify({'success': False, 'error': 'Rainfall should be between 0-3000 mm'}), 400
        
        # Prepare input for prediction
        input_features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]])
        
        if MODEL_AVAILABLE and crop_model and scaler:
            try:
                # Scale the input features
                input_scaled = scaler.transform(input_features)
                
                # Make prediction
                prediction = crop_model.predict(input_scaled)[0]
                prediction_proba = crop_model.predict_proba(input_scaled)[0]
                
                # Get confidence (probability of predicted class)
                confidence = float(max(prediction_proba))
                
                # Map prediction to crop name (assuming the model outputs crop names)
                predicted_crop = str(prediction).lower()
                
                # Get crop information
                crop_info = CROP_INFO.get(predicted_crop, {
                    'emoji': '🌱',
                    'season': 'Varies by region',
                    'duration': '90-150 days',
                    'yield': 'Varies by variety',
                    'market_price': 'Check local markets',
                    'tips': 'Follow good agricultural practices for best results.'
                })
                
                logger.info(f"Crop prediction successful: {predicted_crop} (confidence: {confidence:.2f})")
                
                return jsonify({
                    'success': True,
                    'prediction': {
                        'crop': predicted_crop.title(),
                        'confidence': confidence,
                        'emoji': crop_info['emoji']
                    },
                    'crop_info': crop_info,
                    'input_parameters': {
                        'nitrogen': nitrogen,
                        'phosphorus': phosphorus,
                        'potassium': potassium,
                        'temperature': temperature,
                        'humidity': humidity,
                        'ph': ph,
                        'rainfall': rainfall
                    }
                })
                
            except Exception as model_error:
                logger.error(f"Model prediction error: {model_error}")
                # Fallback to rule-based prediction
                return get_rule_based_prediction(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
        else:
            # Model not available, use rule-based prediction
            logger.info("Using rule-based prediction (model not available)")
            return get_rule_based_prediction(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall)
            
    except Exception as e:
        logger.error(f"Crop prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error during prediction'
        }), 500

def get_rule_based_prediction(nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall):
    """Rule-based crop prediction when ML model is not available"""
    
    # Simple rule-based logic for crop recommendation
    if rainfall > 150 and temperature > 25 and humidity > 70:
        if nitrogen > 80:
            predicted_crop = 'rice'
            confidence = 0.85
        else:
            predicted_crop = 'maize'
            confidence = 0.78
    elif rainfall < 100 and temperature > 20:
        if ph > 7:
            predicted_crop = 'wheat'
            confidence = 0.82
        else:
            predicted_crop = 'potato'
            confidence = 0.75
    elif temperature > 30 and humidity < 60:
        predicted_crop = 'cotton'
        confidence = 0.80
    elif temperature < 25 and rainfall > 100:
        predicted_crop = 'tomato'
        confidence = 0.77
    else:
        # Default recommendation based on balanced conditions
        predicted_crop = 'maize'
        confidence = 0.70
    
    crop_info = CROP_INFO.get(predicted_crop, CROP_INFO['maize'])
    
    return jsonify({
        'success': True,
        'prediction': {
            'crop': predicted_crop.title(),
            'confidence': confidence,
            'emoji': crop_info['emoji']
        },
        'crop_info': crop_info,
        'input_parameters': {
            'nitrogen': nitrogen,
            'phosphorus': phosphorus,
            'potassium': potassium,
            'temperature': temperature,
            'humidity': humidity,
            'ph': ph,
            'rainfall': rainfall
        },
        'note': 'Prediction based on agricultural rules (ML model not available)'
    })

@app.route('/api/upload-image', methods=['POST'])
def upload_image():
    """Upload and convert image to base64 for disease detection"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No image file selected'}), 400
        
        # Read the image file
        image_data = file.read()
        
        # Convert to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        logger.info(f"Image uploaded successfully, size: {len(image_data)} bytes")
        
        return jsonify({
            'success': True,
            'image_base64': image_base64,
            'message': 'Image uploaded successfully'
        })
        
    except Exception as e:
        logger.error(f"Image upload error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to upload image'
        }), 500

@app.route('/api/disease-detection', methods=['POST'])
def detect_disease():
    """Detect plant disease from base64 image (mock implementation)"""
    try:
        data = request.json
        
        if not data or 'image_base64' not in data:
            return jsonify({'success': False, 'error': 'No image data provided'}), 400
        
        image_base64 = data['image_base64']
        
        if not image_base64:
            return jsonify({'success': False, 'error': 'Empty image data'}), 400
        
        logger.info("Processing disease detection request")
        
        # Mock disease detection (replace with actual ML model when available)
        diseases = [
            {
                'name': 'Leaf Blight',
                'confidence': 0.85,
                'severity': 'Moderate',
                'emoji': '🍃',
                'description': 'A common fungal disease affecting leaves, causing brown spots and yellowing.',
                'treatment': 'Apply copper-based fungicide spray every 7-10 days. Remove affected leaves immediately.',
                'prevention': 'Ensure proper air circulation, avoid overhead watering, and maintain field hygiene.'
            },
            {
                'name': 'Powdery Mildew',
                'confidence': 0.78,
                'severity': 'Mild',
                'emoji': '🌿',
                'description': 'White powdery coating on leaves and stems, reducing photosynthesis.',
                'treatment': 'Spray with neem oil or sulfur-based fungicide. Improve air circulation.',
                'prevention': 'Plant resistant varieties, avoid overcrowding, and water at soil level.'
            },
            {
                'name': 'Bacterial Wilt',
                'confidence': 0.92,
                'severity': 'Severe',
                'emoji': '🦠',
                'description': 'Bacterial infection causing wilting and yellowing of plants.',
                'treatment': 'Remove infected plants immediately. Apply copper sulfate solution to soil.',
                'prevention': 'Use disease-free seeds, practice crop rotation, and maintain proper drainage.'
            },
            {
                'name': 'Healthy Plant',
                'confidence': 0.95,
                'severity': 'None',
                'emoji': '✅',
                'description': 'Plant appears healthy with no visible signs of disease.',
                'treatment': 'No treatment needed. Continue regular care and monitoring.',
                'prevention': 'Maintain current care practices and monitor for any changes.'
            }
        ]
        
        # Randomly select a disease for demonstration
        detected_disease = random.choice(diseases)
        
        logger.info(f"Disease detection result: {detected_disease['name']} (confidence: {detected_disease['confidence']:.2f})")
        
        return jsonify({
            'success': True,
            'disease': {
                'name': detected_disease['name'],
                'confidence': detected_disease['confidence'],
                'severity': detected_disease['severity'],
                'emoji': detected_disease['emoji']
            },
            'diagnosis': {
                'description': detected_disease['description'],
                'treatment': detected_disease['treatment'],
                'prevention': detected_disease['prevention']
            }
        })
        
    except Exception as e:
        logger.error(f"Disease detection error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to detect disease'
        }), 500

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
            'requests': REQUESTS_AVAILABLE,
            'crop_model': MODEL_AVAILABLE
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
            'knowledge_base': True,
            'crop_prediction': MODEL_AVAILABLE
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

@app.route('/api/crop-calendar', methods=['GET'])
def crop_calendar():
    """Get comprehensive crop calendar for Smart India Hackathon"""
    try:
        current_month = datetime.datetime.now().month
        
        # Comprehensive crop calendar data for Indian agriculture
        crop_calendar_data = {
            'kharif_crops': {
                'season': 'Kharif (June-October)',
                'description': 'Monsoon crops grown during rainy season',
                'crops': [
                    {
                        'name': 'Rice',
                        'emoji': '🌾',
                        'sowing_months': [6, 7],
                        'harvesting_months': [10, 11],
                        'duration_days': 120,
                        'water_requirement': 'High',
                        'soil_type': 'Clay loam',
                        'major_states': ['West Bengal', 'Punjab', 'Uttar Pradesh', 'Andhra Pradesh'],
                        'yield_per_hectare': '3-4 tonnes',
                        'market_price_range': '₹2000-2500/quintal'
                    },
                    {
                        'name': 'Cotton',
                        'emoji': '🌿',
                        'sowing_months': [5, 6],
                        'harvesting_months': [10, 11, 12],
                        'duration_days': 180,
                        'water_requirement': 'Medium',
                        'soil_type': 'Black cotton soil',
                        'major_states': ['Gujarat', 'Maharashtra', 'Telangana', 'Punjab'],
                        'yield_per_hectare': '1.5-2 tonnes',
                        'market_price_range': '₹5500-6500/quintal'
                    },
                    {
                        'name': 'Sugarcane',
                        'emoji': '🎋',
                        'sowing_months': [2, 3, 4],
                        'harvesting_months': [12, 1, 2, 3],
                        'duration_days': 365,
                        'water_requirement': 'Very High',
                        'soil_type': 'Rich loamy soil',
                        'major_states': ['Uttar Pradesh', 'Maharashtra', 'Karnataka', 'Tamil Nadu'],
                        'yield_per_hectare': '70-80 tonnes',
                        'market_price_range': '₹300-350/quintal'
                    }
                ]
            },
            'rabi_crops': {
                'season': 'Rabi (November-April)',
                'description': 'Winter crops grown during dry season',
                'crops': [
                    {
                        'name': 'Wheat',
                        'emoji': '🌾',
                        'sowing_months': [11, 12],
                        'harvesting_months': [3, 4],
                        'duration_days': 120,
                        'water_requirement': 'Medium',
                        'soil_type': 'Well-drained loamy soil',
                        'major_states': ['Uttar Pradesh', 'Punjab', 'Haryana', 'Madhya Pradesh'],
                        'yield_per_hectare': '3-4 tonnes',
                        'market_price_range': '₹2100-2400/quintal'
                    },
                    {
                        'name': 'Mustard',
                        'emoji': '🌻',
                        'sowing_months': [10, 11],
                        'harvesting_months': [2, 3],
                        'duration_days': 120,
                        'water_requirement': 'Low',
                        'soil_type': 'Sandy loam',
                        'major_states': ['Rajasthan', 'Haryana', 'Uttar Pradesh', 'West Bengal'],
                        'yield_per_hectare': '1-1.5 tonnes',
                        'market_price_range': '₹4500-5500/quintal'
                    }
                ]
            },
            'zaid_crops': {
                'season': 'Zaid (March-June)',
                'description': 'Summer crops grown with irrigation',
                'crops': [
                    {
                        'name': 'Watermelon',
                        'emoji': '🍉',
                        'sowing_months': [2, 3],
                        'harvesting_months': [5, 6],
                        'duration_days': 90,
                        'water_requirement': 'High',
                        'soil_type': 'Sandy loam',
                        'major_states': ['Uttar Pradesh', 'Rajasthan', 'Punjab', 'Haryana'],
                        'yield_per_hectare': '20-25 tonnes',
                        'market_price_range': '₹800-1500/quintal'
                    },
                    {
                        'name': 'Fodder Maize',
                        'emoji': '🌽',
                        'sowing_months': [3, 4],
                        'harvesting_months': [6, 7],
                        'duration_days': 90,
                        'water_requirement': 'Medium',
                        'soil_type': 'Well-drained soil',
                        'major_states': ['Punjab', 'Haryana', 'Uttar Pradesh', 'Bihar'],
                        'yield_per_hectare': '40-50 tonnes',
                        'market_price_range': '₹1200-1800/quintal'
                    }
                ]
            }
        }
        
        # Get current month activities
        current_activities = []
        for season_data in crop_calendar_data.values():
            for crop in season_data['crops']:
                if current_month in crop['sowing_months']:
                    current_activities.append({
                        'activity': 'Sowing',
                        'crop': crop['name'],
                        'emoji': crop['emoji'],
                        'priority': 'High'
                    })
                elif current_month in crop['harvesting_months']:
                    current_activities.append({
                        'activity': 'Harvesting',
                        'crop': crop['name'],
                        'emoji': crop['emoji'],
                        'priority': 'High'
                    })
        
        # Smart recommendations based on current month
        smart_recommendations = []
        if current_month in [6, 7, 8]:  # Monsoon season
            smart_recommendations.extend([
                "🌧️ Monitor rainfall levels for optimal rice cultivation",
                "🦠 Watch for fungal diseases due to high humidity",
                "💧 Ensure proper drainage to prevent waterlogging"
            ])
        elif current_month in [11, 12, 1]:  # Winter season
            smart_recommendations.extend([
                "❄️ Protect crops from frost damage",
                "💧 Schedule irrigation as per crop water requirements",
                "🌱 Apply balanced fertilizers for winter crops"
            ])
        elif current_month in [3, 4, 5]:  # Summer season
            smart_recommendations.extend([
                "☀️ Increase irrigation frequency due to high temperatures",
                "🌿 Use mulching to conserve soil moisture",
                "🕐 Schedule farm activities during cooler hours"
            ])
        
        return jsonify({
            'success': True,
            'current_month': current_month,
            'current_activities': current_activities,
            'smart_recommendations': smart_recommendations,
            'crop_calendar': crop_calendar_data,
            'generated_for': 'Smart India Hackathon 2025',
            'last_updated': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Crop calendar error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch crop calendar data'
        }), 500

@app.route('/api/farmer-advisory', methods=['POST'])
def farmer_advisory():
    """AI-powered farmer advisory system for Smart India Hackathon"""
    try:
        data = request.json
        location = data.get('location', 'India')
        crop_type = data.get('crop_type', 'general')
        issue_type = data.get('issue_type', 'general')
        
        # Smart advisory responses based on issue type
        advisory_responses = {
            'pest_control': [
                "🐛 Integrated Pest Management (IPM) approach is recommended",
                "🌿 Use neem-based organic pesticides as first line of defense",
                "🕷️ Encourage beneficial insects like ladybugs and spiders",
                "📅 Regular monitoring and early detection is crucial",
                "💧 Avoid over-watering which can attract pests"
            ],
            'disease_management': [
                "🦠 Ensure proper crop rotation to break disease cycles",
                "💨 Maintain good air circulation between plants",
                "🌱 Use disease-resistant varieties when available",
                "🧪 Apply copper-based fungicides for fungal diseases",
                "🗑️ Remove and destroy infected plant material immediately"
            ],
            'soil_health': [
                "🧪 Conduct regular soil testing for pH and nutrients",
                "🌿 Add organic matter like compost and vermicompost",
                "🔄 Practice crop rotation to maintain soil fertility",
                "🌱 Use cover crops during fallow periods",
                "⚖️ Balance NPK ratios based on crop requirements"
            ],
            'water_management': [
                "💧 Implement drip irrigation for water efficiency",
                "🕐 Water during early morning or evening hours",
                "🌿 Use mulching to reduce water evaporation",
                "📊 Monitor soil moisture levels regularly",
                "🌧️ Harvest rainwater for irrigation purposes"
            ],
            'fertilizer_management': [
                "🧪 Apply fertilizers based on soil test recommendations",
                "📅 Use split application for nitrogen fertilizers",
                "🌿 Combine organic and inorganic fertilizers",
                "⏰ Apply fertilizers at the right growth stages",
                "💧 Ensure adequate moisture for nutrient uptake"
            ]
        }
        
        # Get relevant advisory
        advisory = advisory_responses.get(issue_type, [
            "🌾 Follow good agricultural practices for better yields",
            "📚 Stay updated with latest farming techniques",
            "🤝 Connect with local agricultural extension officers",
            "📱 Use technology for precision farming",
            "🌍 Consider sustainable farming practices"
        ])
        
        # Add location-specific advice
        location_advice = []
        if 'punjab' in location.lower() or 'haryana' in location.lower():
            location_advice.append("🌾 Focus on wheat-rice rotation system")
            location_advice.append("💧 Manage groundwater depletion issues")
        elif 'maharashtra' in location.lower():
            location_advice.append("🌿 Consider cotton and sugarcane cultivation")
            location_advice.append("🌧️ Plan for monsoon variability")
        elif 'kerala' in location.lower():
            location_advice.append("🥥 Coconut and spice cultivation is ideal")
            location_advice.append("🌧️ Manage high humidity related diseases")
        
        return jsonify({
            'success': True,
            'advisory': advisory,
            'location_specific': location_advice,
            'crop_type': crop_type,
            'issue_type': issue_type,
            'generated_by': 'KrishiMitra AI - Smart India Hackathon 2025',
            'timestamp': datetime.datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Farmer advisory error: {e}")
        return jsonify({
            'success': False,
            'error': 'Failed to generate farmer advisory'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting KrishiMitra API on port {port}")
    logger.info(f"Gemini API configured: {bool(GEMINI_API_KEY)}")
    logger.info(f"Weather API configured: {bool(WEATHER_API_KEY)}")
    app.run(host='0.0.0.0', port=port, debug=False)
