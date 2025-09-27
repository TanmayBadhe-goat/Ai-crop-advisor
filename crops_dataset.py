# Comprehensive Crops Dataset for KrishiMitra - Smart India Hackathon
# This file contains detailed information about various crops grown in India

# Main crop information database
CROP_INFO = {
    # Cereals
    'rice': {
        'emoji': '🌾',
        'category': 'Cereal',
        'season': 'Kharif (June-October)',
        'duration': '120-150 days',
        'yield': '40-60 quintals/hectare',
        'market_price': '₹2000-2500/quintal',
        'water_requirement': 'High (1200-1500mm)',
        'soil_type': 'Clay loam, well-drained',
        'temperature': '20-35°C',
        'tips': 'Maintain 2-5cm water level. Apply fertilizers in split doses. Harvest when 80% grains turn golden.'
    },
    'wheat': {
        'emoji': '🌾',
        'category': 'Cereal',
        'season': 'Rabi (November-April)',
        'duration': '120-140 days',
        'yield': '35-50 quintals/hectare',
        'market_price': '₹2100-2400/quintal',
        'water_requirement': 'Medium (450-650mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '15-25°C',
        'tips': 'Sow in well-prepared field. Irrigate 4-6 times. Apply nitrogen in 3 split doses.'
    },
    'maize': {
        'emoji': '🌽',
        'category': 'Cereal',
        'season': 'Kharif & Rabi',
        'duration': '90-120 days',
        'yield': '50-80 quintals/hectare',
        'market_price': '₹1800-2200/quintal',
        'water_requirement': 'Medium (500-800mm)',
        'soil_type': 'Well-drained fertile soil',
        'temperature': '21-27°C',
        'tips': 'Plant in rows with proper spacing. Apply balanced fertilizers. Control stem borer and fall armyworm.'
    },
    'bajra': {
        'emoji': '🌾',
        'category': 'Millet',
        'season': 'Kharif (June-September)',
        'duration': '75-90 days',
        'yield': '10-15 quintals/hectare',
        'market_price': '₹2000-2500/quintal',
        'water_requirement': 'Low (350-500mm)',
        'soil_type': 'Sandy loam, drought tolerant',
        'temperature': '25-35°C',
        'tips': 'Drought resistant crop. Suitable for arid regions. Minimal water requirement.'
    },
    'jowar': {
        'emoji': '🌾',
        'category': 'Millet',
        'season': 'Kharif & Rabi',
        'duration': '100-120 days',
        'yield': '15-25 quintals/hectare',
        'market_price': '₹2200-2800/quintal',
        'water_requirement': 'Low (400-600mm)',
        'soil_type': 'Well-drained black soil',
        'temperature': '26-30°C',
        'tips': 'Heat and drought tolerant. Good for semi-arid regions. Rich in protein.'
    },
    'ragi': {
        'emoji': '🌾',
        'category': 'Millet',
        'season': 'Kharif (June-October)',
        'duration': '120-150 days',
        'yield': '15-20 quintals/hectare',
        'market_price': '₹3000-4000/quintal',
        'water_requirement': 'Medium (500-750mm)',
        'soil_type': 'Red loamy soil',
        'temperature': '20-30°C',
        'tips': 'High calcium content. Suitable for hilly regions. Good for health-conscious consumers.'
    },
    
    # Pulses
    'chickpea': {
        'emoji': '🫘',
        'category': 'Pulse',
        'season': 'Rabi (October-March)',
        'duration': '90-120 days',
        'yield': '15-20 quintals/hectare',
        'market_price': '₹4500-6000/quintal',
        'water_requirement': 'Low (300-400mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '20-30°C',
        'tips': 'Nitrogen fixing crop. Avoid waterlogging. Good protein source.'
    },
    'lentil': {
        'emoji': '🫘',
        'category': 'Pulse',
        'season': 'Rabi (October-March)',
        'duration': '95-110 days',
        'yield': '8-12 quintals/hectare',
        'market_price': '₹5000-7000/quintal',
        'water_requirement': 'Low (300-400mm)',
        'soil_type': 'Well-drained sandy loam',
        'temperature': '18-30°C',
        'tips': 'Cool weather crop. Rich in protein. Good for crop rotation.'
    },
    'moong': {
        'emoji': '🫘',
        'category': 'Pulse',
        'season': 'Kharif & Summer',
        'duration': '60-90 days',
        'yield': '8-12 quintals/hectare',
        'market_price': '₹6000-8000/quintal',
        'water_requirement': 'Medium (350-400mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '25-35°C',
        'tips': 'Short duration crop. Good for intercropping. High protein content.'
    },
    
    # Cash Crops
    'sugarcane': {
        'emoji': '🎋',
        'category': 'Cash Crop',
        'season': 'Year-round',
        'duration': '12-18 months',
        'yield': '800-1200 quintals/hectare',
        'market_price': '₹300-350/quintal',
        'water_requirement': 'Very High (1500-2500mm)',
        'soil_type': 'Rich loamy soil',
        'temperature': '20-30°C',
        'tips': 'Plant healthy seed cane. Maintain adequate moisture. Harvest at proper maturity for maximum sugar content.'
    },
    'cotton': {
        'emoji': '🌿',
        'category': 'Cash Crop',
        'season': 'Kharif (April-October)',
        'duration': '180-200 days',
        'yield': '15-25 quintals/hectare',
        'market_price': '₹5500-6500/quintal',
        'water_requirement': 'Medium (500-800mm)',
        'soil_type': 'Black cotton soil',
        'temperature': '21-30°C',
        'tips': 'Deep ploughing required. Monitor for bollworm. Pick cotton when bolls are fully opened.'
    },
    'tobacco': {
        'emoji': '🍃',
        'category': 'Cash Crop',
        'season': 'Rabi (November-April)',
        'duration': '120-150 days',
        'yield': '20-25 quintals/hectare',
        'market_price': '₹8000-12000/quintal',
        'water_requirement': 'Medium (400-600mm)',
        'soil_type': 'Well-drained sandy loam',
        'temperature': '20-30°C',
        'tips': 'Requires careful curing process. High value crop but health concerns.'
    },
    'jute': {
        'emoji': '🌿',
        'category': 'Fiber Crop',
        'season': 'Kharif (April-July)',
        'duration': '120-150 days',
        'yield': '25-30 quintals/hectare',
        'market_price': '₹4000-5000/quintal',
        'water_requirement': 'High (1000-1200mm)',
        'soil_type': 'Alluvial soil',
        'temperature': '25-35°C',
        'tips': 'Requires high humidity. Eco-friendly fiber. Good for wet regions.'
    },
    
    # Oilseeds
    'mustard': {
        'emoji': '🌻',
        'category': 'Oilseed',
        'season': 'Rabi (October-March)',
        'duration': '120-150 days',
        'yield': '12-18 quintals/hectare',
        'market_price': '₹4500-5500/quintal',
        'water_requirement': 'Low (300-400mm)',
        'soil_type': 'Sandy loam',
        'temperature': '15-25°C',
        'tips': 'Cool weather crop. Good oil content. Suitable for northern plains.'
    },
    'groundnut': {
        'emoji': '🥜',
        'category': 'Oilseed',
        'season': 'Kharif & Rabi',
        'duration': '100-130 days',
        'yield': '20-25 quintals/hectare',
        'market_price': '₹5000-6000/quintal',
        'water_requirement': 'Medium (500-750mm)',
        'soil_type': 'Well-drained sandy loam',
        'temperature': '20-30°C',
        'tips': 'Requires calcium for pod development. Good protein and oil source.'
    },
    'soybean': {
        'emoji': '🫘',
        'category': 'Oilseed',
        'season': 'Kharif (June-October)',
        'duration': '90-120 days',
        'yield': '15-20 quintals/hectare',
        'market_price': '₹3500-4500/quintal',
        'water_requirement': 'Medium (450-700mm)',
        'soil_type': 'Well-drained black soil',
        'temperature': '20-30°C',
        'tips': 'High protein content. Good for crop rotation. Nitrogen fixing.'
    },
    
    # Plantation Crops
    'tea': {
        'emoji': '🍃',
        'category': 'Plantation',
        'season': 'Year-round',
        'duration': '3-5 years to mature',
        'yield': '1500-2000 kg/hectare',
        'market_price': '₹200-500/kg',
        'water_requirement': 'High (1200-1500mm)',
        'soil_type': 'Well-drained acidic soil',
        'temperature': '20-30°C',
        'tips': 'Requires high altitude. Regular pruning needed. Multiple harvests per year.'
    },
    'coffee': {
        'emoji': '☕',
        'category': 'Plantation',
        'season': 'Year-round',
        'duration': '3-4 years to mature',
        'yield': '800-1200 kg/hectare',
        'market_price': '₹300-600/kg',
        'water_requirement': 'Medium (1000-1500mm)',
        'soil_type': 'Well-drained red soil',
        'temperature': '15-25°C',
        'tips': 'Shade grown crop. Requires cool climate. High value export crop.'
    },
    
    # Spices
    'ginger': {
        'emoji': '🫚',
        'category': 'Spice',
        'season': 'Kharif (April-July)',
        'duration': '8-10 months',
        'yield': '150-200 quintals/hectare',
        'market_price': '₹2000-4000/quintal',
        'water_requirement': 'High (1000-1500mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '25-30°C',
        'tips': 'Shade loving crop. High value spice. Requires good drainage.'
    },
    'saffron': {
        'emoji': '🌸',
        'category': 'Spice',
        'season': 'Rabi (October-November)',
        'duration': '150-180 days',
        'yield': '8-12 kg/hectare',
        'market_price': '₹250000-400000/kg',
        'water_requirement': 'Low (300-400mm)',
        'soil_type': 'Well-drained sandy loam',
        'temperature': '15-20°C',
        'tips': 'World\'s most expensive spice. Requires cool climate. Hand harvested.'
    },
    
    # Fruits
    'mango': {
        'emoji': '🥭',
        'category': 'Fruit',
        'season': 'Year-round',
        'duration': '3-5 years to bear',
        'yield': '100-200 quintals/hectare',
        'market_price': '₹2000-5000/quintal',
        'water_requirement': 'Medium (600-1200mm)',
        'soil_type': 'Well-drained deep soil',
        'temperature': '24-30°C',
        'tips': 'King of fruits. Requires hot climate. Good export potential.'
    },
    'banana': {
        'emoji': '🍌',
        'category': 'Fruit',
        'season': 'Year-round',
        'duration': '12-15 months',
        'yield': '400-600 quintals/hectare',
        'market_price': '₹1200-1800/quintal',
        'water_requirement': 'High (1200-2000mm)',
        'soil_type': 'Rich loamy soil',
        'temperature': '26-30°C',
        'tips': 'Plant tissue culture plants. Maintain adequate drainage. Remove excess suckers regularly.'
    },
    
    # Vegetables
    'tomato': {
        'emoji': '🍅',
        'category': 'Vegetable',
        'season': 'Kharif & Rabi',
        'duration': '120-150 days',
        'yield': '400-600 quintals/hectare',
        'market_price': '₹1000-2000/quintal',
        'water_requirement': 'Medium (400-600mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '20-25°C',
        'tips': 'Use disease-resistant varieties. Provide support to plants. Regular pruning increases yield.'
    },
    'potato': {
        'emoji': '🥔',
        'category': 'Vegetable',
        'season': 'Rabi (October-March)',
        'duration': '90-120 days',
        'yield': '250-400 quintals/hectare',
        'market_price': '₹800-1500/quintal',
        'water_requirement': 'Medium (500-700mm)',
        'soil_type': 'Well-drained sandy loam',
        'temperature': '15-20°C',
        'tips': 'Plant certified seed potatoes. Earth up regularly. Control late blight disease.'
    },
    'onion': {
        'emoji': '🧅',
        'category': 'Vegetable',
        'season': 'Rabi & Kharif',
        'duration': '120-150 days',
        'yield': '200-400 quintals/hectare',
        'market_price': '₹1000-3000/quintal',
        'water_requirement': 'Medium (350-550mm)',
        'soil_type': 'Well-drained fertile soil',
        'temperature': '15-25°C',
        'tips': 'Requires well-prepared seedbed. Good storage crop. High market demand.'
    },
    'brinjal': {
        'emoji': '🍆',
        'category': 'Vegetable',
        'season': 'Year-round',
        'duration': '120-150 days',
        'yield': '250-400 quintals/hectare',
        'market_price': '₹800-1500/quintal',
        'water_requirement': 'Medium (600-800mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '22-32°C',
        'tips': 'Warm season crop. Control fruit and shoot borer. Regular harvesting needed.'
    },
    'cabbage': {
        'emoji': '🥬',
        'category': 'Vegetable',
        'season': 'Rabi (October-February)',
        'duration': '90-120 days',
        'yield': '400-600 quintals/hectare',
        'market_price': '₹500-1200/quintal',
        'water_requirement': 'Medium (400-500mm)',
        'soil_type': 'Well-drained fertile soil',
        'temperature': '15-20°C',
        'tips': 'Cool season crop. Requires consistent moisture. Good source of vitamin C.'
    },
    'cauliflower': {
        'emoji': '🥦',
        'category': 'Vegetable',
        'season': 'Rabi (October-February)',
        'duration': '100-120 days',
        'yield': '200-300 quintals/hectare',
        'market_price': '₹800-1500/quintal',
        'water_requirement': 'Medium (400-500mm)',
        'soil_type': 'Well-drained fertile soil',
        'temperature': '15-20°C',
        'tips': 'Cool season crop. Protect curds from sun. Rich in vitamin C and minerals.'
    },
    'peas': {
        'emoji': '🟢',
        'category': 'Vegetable',
        'season': 'Rabi (October-February)',
        'duration': '90-110 days',
        'yield': '80-120 quintals/hectare',
        'market_price': '₹2000-4000/quintal',
        'water_requirement': 'Medium (300-400mm)',
        'soil_type': 'Well-drained loamy soil',
        'temperature': '10-18°C',
        'tips': 'Cool season crop. Nitrogen fixing. Good protein source.'
    },
    'carrot': {
        'emoji': '🥕',
        'category': 'Vegetable',
        'season': 'Rabi (October-January)',
        'duration': '90-120 days',
        'yield': '200-300 quintals/hectare',
        'market_price': '₹1000-2000/quintal',
        'water_requirement': 'Medium (400-500mm)',
        'soil_type': 'Deep sandy loam',
        'temperature': '16-20°C',
        'tips': 'Cool season crop. Requires deep soil. Rich in vitamin A.'
    },
    'capsicum': {
        'emoji': '🫑',
        'category': 'Vegetable',
        'season': 'Year-round',
        'duration': '120-150 days',
        'yield': '250-400 quintals/hectare',
        'market_price': '₹1500-3000/quintal',
        'water_requirement': 'Medium (600-800mm)',
        'soil_type': 'Well-drained fertile soil',
        'temperature': '20-25°C',
        'tips': 'Warm season crop. High value vegetable. Requires protected cultivation in some regions.'
    }
}

# Agricultural Knowledge Base for Fallback (Enhanced)
AGRICULTURAL_KNOWLEDGE = {
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
    'maize': {
        'keywords': ['maize', 'corn', 'makka', 'मक्का'],
        'responses': [
            "🌽 Maize grows in both Kharif and Rabi seasons. Requires well-drained soil and 21-27°C temperature.",
            "🌽 For maize: Apply 120kg N, 60kg P2O5, 40kg K2O per hectare. Plant with 60cm row spacing.",
            "🌽 Control stem borer and fall armyworm. Harvest when kernels are hard and moisture is 15-20%."
        ]
    },
    'cotton': {
        'keywords': ['cotton', 'kapas', 'कपास'],
        'responses': [
            "🌿 Cotton is a Kharif crop requiring 180-200 frost-free days. Plant in April-June with black cotton soil.",
            "🌿 For cotton: Apply 120kg N, 60kg P2O5, 30kg K2O per hectare. Maintain soil moisture at 70-80%.",
            "🌿 Monitor for bollworm, whitefly, and pink bollworm. Pick cotton when bolls are fully opened."
        ]
    },
    'tomato': {
        'keywords': ['tomato', 'tamatar', 'टमाटर'],
        'responses': [
            "🍅 Tomatoes grow year-round with proper care. Require well-drained soil and 20-25°C temperature.",
            "🍅 For tomatoes: Apply 100kg N, 50kg P2O5, 50kg K2O per hectare. Provide support stakes.",
            "🍅 Control early blight, late blight, and fruit borer. Harvest when fruits are firm and red."
        ]
    },
    'potato': {
        'keywords': ['potato', 'aloo', 'आलू'],
        'responses': [
            "🥔 Potatoes are Rabi crops planted in October-November. Require cool weather and well-drained soil.",
            "🥔 For potatoes: Apply 120kg N, 60kg P2O5, 60kg K2O per hectare. Earth up regularly.",
            "🥔 Control late blight and potato tuber moth. Harvest when plants turn yellow and dry."
        ]
    },
    'sugarcane': {
        'keywords': ['sugarcane', 'ganna', 'गन्ना'],
        'responses': [
            "🎋 Sugarcane is a long-duration crop (12-18 months). Requires high water and fertile soil.",
            "🎋 Plant healthy seed cane with 2-3 buds. Maintain adequate moisture throughout growing period.",
            "🎋 Harvest at proper maturity for maximum sugar content. Good for sugar and jaggery production."
        ]
    },
    'fertilizer': {
        'keywords': ['fertilizer', 'khad', 'खाद', 'urea', 'dap', 'npk'],
        'responses': [
            "🌱 NPK fertilizers: N for leaf growth, P for roots/flowers, K for disease resistance. Test soil before applying.",
            "🌱 Organic fertilizers: Compost, vermicompost, green manure improve soil health long-term.",
            "🌱 Apply fertilizers in split doses: 1/3 at sowing, 1/3 at vegetative stage, 1/3 at flowering."
        ]
    },
    'irrigation': {
        'keywords': ['irrigation', 'water', 'watering', 'सिंचाई', 'पानी'],
        'responses': [
            "💧 Drip irrigation saves 30-50% water compared to flood irrigation. Best for water-scarce areas.",
            "💧 Water crops early morning or evening to reduce evaporation. Check soil moisture regularly.",
            "💧 Critical irrigation stages: germination, flowering, and grain filling. Avoid waterlogging."
        ]
    },
    'pest_control': {
        'keywords': ['pest', 'insect', 'bug', 'कीट', 'disease', 'बीमारी'],
        'responses': [
            "🐛 Use IPM (Integrated Pest Management): biological, cultural, and chemical methods together.",
            "🐛 Neem oil is effective against aphids, whiteflies, and thrips. Spray during cooler hours.",
            "🐛 Monitor crops regularly. Use pheromone traps and beneficial insects like ladybugs."
        ]
    },
    'soil': {
        'keywords': ['soil', 'मिट्टी', 'ph', 'nutrients', 'testing'],
        'responses': [
            "🌱 Test soil pH annually. Most crops prefer 6.0-7.5 pH. Add lime to increase, sulfur to decrease pH.",
            "🌱 Soil health indicators: organic matter, water retention, and microbial activity.",
            "🌱 Add compost and crop rotation to improve soil structure and fertility naturally."
        ]
    },
    'weather': {
        'keywords': ['weather', 'rain', 'temperature', 'मौसम', 'climate'],
        'responses': [
            "🌤️ Monitor weather forecasts for irrigation and pest management decisions.",
            "🌤️ Protect crops from extreme weather: use mulching, shade nets, and windbreaks.",
            "🌤️ Adjust planting dates based on monsoon predictions and temperature patterns."
        ]
    },
    'organic': {
        'keywords': ['organic', 'natural', 'जैविक', 'compost', 'vermicompost'],
        'responses': [
            "🌿 Organic farming uses natural inputs: compost, biofertilizers, and biopesticides.",
            "🌿 Vermicompost provides slow-release nutrients and improves soil structure.",
            "🌿 Crop rotation and green manuring are key practices in organic farming."
        ]
    },
    'seeds': {
        'keywords': ['seed', 'variety', 'बीज', 'planting', 'sowing'],
        'responses': [
            "🌱 Use certified seeds from authorized dealers. Check germination rate before sowing.",
            "🌱 Treat seeds with fungicide or bioagents to prevent soil-borne diseases.",
            "🌱 Choose varieties suitable for your region's climate and soil conditions."
        ]
    }
}

# Crop categories for easy filtering
CROP_CATEGORIES = {
    'cereals': ['rice', 'wheat', 'maize', 'bajra', 'jowar', 'ragi'],
    'pulses': ['chickpea', 'lentil', 'moong'],
    'cash_crops': ['sugarcane', 'cotton', 'tobacco', 'jute'],
    'oilseeds': ['mustard', 'groundnut', 'soybean'],
    'plantation': ['tea', 'coffee'],
    'spices': ['ginger', 'saffron'],
    'fruits': ['mango', 'banana'],
    'vegetables': ['tomato', 'potato', 'onion', 'brinjal', 'cabbage', 'cauliflower', 'peas', 'carrot', 'capsicum']
}

# Season-wise crop classification
SEASONAL_CROPS = {
    'kharif': ['rice', 'maize', 'cotton', 'sugarcane', 'bajra', 'jowar', 'ragi', 'moong', 'soybean', 'jute', 'ginger'],
    'rabi': ['wheat', 'chickpea', 'lentil', 'mustard', 'potato', 'onion', 'cabbage', 'cauliflower', 'peas', 'carrot', 'tobacco', 'saffron'],
    'zaid': ['maize', 'moong', 'groundnut'],
    'perennial': ['sugarcane', 'tea', 'coffee', 'mango', 'banana']
}

def get_crop_info(crop_name):
    """Get detailed information about a specific crop"""
    return CROP_INFO.get(crop_name.lower(), None)

def get_crops_by_category(category):
    """Get all crops in a specific category"""
    return CROP_CATEGORIES.get(category.lower(), [])

def get_crops_by_season(season):
    """Get all crops for a specific season"""
    return SEASONAL_CROPS.get(season.lower(), [])

def get_all_crop_names():
    """Get list of all available crop names"""
    return list(CROP_INFO.keys())

def search_crops(query):
    """Search crops by name or category"""
    query = query.lower()
    results = []
    
    for crop_name, crop_data in CROP_INFO.items():
        if query in crop_name or query in crop_data.get('category', '').lower():
            results.append(crop_name)
    
    return results
