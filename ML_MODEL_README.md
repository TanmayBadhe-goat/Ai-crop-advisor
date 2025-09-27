# KrishiMitra ML Model Training - Smart India Hackathon

This directory contains the machine learning model training scripts for KrishiMitra, supporting **31 different crops** with comprehensive agricultural data.

## 🌾 Supported Crops

### Cereals (6)
- Rice, Wheat, Maize, Bajra (Pearl Millet), Jowar (Sorghum), Ragi (Finger Millet)

### Pulses (3)
- Chickpea, Lentil, Moong

### Cash Crops (4)
- Sugarcane, Cotton, Tobacco, Jute

### Oilseeds (3)
- Mustard, Groundnut, Soybean

### Plantation Crops (2)
- Tea, Coffee

### Spices (2)
- Ginger, Saffron

### Fruits (2)
- Mango, Banana

### Vegetables (9)
- Tomato, Potato, Onion, Brinjal (Eggplant), Cabbage, Cauliflower, Peas, Carrot, Capsicum (Bell Pepper)

## 📁 Files Overview

### Training Scripts
- **`quick_train.py`** - Fast training for immediate use (recommended for demo)
- **`train_model.py`** - Comprehensive training with detailed analysis
- **`validate_model.py`** - Model validation and testing

### Data Files
- **`crops_dataset.py`** - Comprehensive crop database
- **`backend/data/model.pkl`** - Trained ML model
- **`backend/data/scaler.pkl`** - Feature scaler

## 🚀 Quick Start

### Option 1: Quick Training (Recommended for SIH Demo)
```bash
python quick_train.py
```
- ⚡ Fast execution (~30 seconds)
- 🎯 97%+ accuracy
- 📊 3,100 training samples
- ✅ Ready for demo immediately

### Option 2: Comprehensive Training
```bash
python train_model.py
```
- 🔬 Detailed analysis and reporting
- 📈 Higher accuracy with more samples
- 📋 Feature importance analysis
- 🧪 Comprehensive testing

### Validation
```bash
python validate_model.py
```
- ✅ Validates model coverage
- 🧪 Tests predictions
- 📊 Generates detailed reports

## 🎯 Model Features

### Input Parameters (7)
1. **Nitrogen** (0-300 kg/hectare)
2. **Phosphorus** (0-150 kg/hectare)
3. **Potassium** (0-200 kg/hectare)
4. **Temperature** (5-45°C)
5. **Humidity** (30-95%)
6. **pH** (4.0-9.0)
7. **Rainfall** (0-3000mm)

### Output
- **Crop Recommendation** with confidence score
- **Detailed Crop Information** (season, duration, yield, market price, tips)

## 📊 Model Performance

- **Accuracy**: 97%+ on training data
- **Crops Supported**: 31 different crops
- **Training Samples**: 3,100+ samples
- **Model Type**: Random Forest Classifier
- **Features**: 7 agricultural parameters

## 🔗 API Integration

The trained model integrates with the KrishiMitra API:

```python
# API Endpoint
POST /api/predict

# Request Body
{
  "nitrogen": 100,
  "phosphorus": 60,
  "potassium": 50,
  "temperature": 25,
  "humidity": 80,
  "ph": 6.0,
  "rainfall": 200
}

# Response
{
  "success": true,
  "prediction": {
    "crop": "Rice",
    "confidence": 0.95,
    "emoji": "🌾"
  },
  "crop_info": {
    "season": "Kharif (June-October)",
    "duration": "120-150 days",
    "yield": "40-60 quintals/hectare",
    "market_price": "₹2000-2500/quintal",
    "tips": "Maintain 2-5cm water level..."
  }
}
```

## 🏆 Smart India Hackathon Features

### Technical Highlights
- **Comprehensive Dataset**: 31 crops covering all major Indian agriculture
- **Scientific Accuracy**: Based on real agricultural parameters
- **Production Ready**: Robust error handling and validation
- **Scalable Architecture**: Easy to add more crops

### Demo Points for Judges
1. **"Real ML Implementation"** - Not just a UI mockup
2. **"Comprehensive Coverage"** - Supports major Indian crops
3. **"Scientific Approach"** - Based on actual agricultural parameters
4. **"Production Quality"** - Proper validation and testing

## 🛠️ Development Workflow

### For SIH Demo
1. Run `python quick_train.py` to generate model
2. Test with `python validate_model.py`
3. Deploy to Railway backend
4. Demo crop recommendations in app

### For Production
1. Run `python train_model.py` for comprehensive training
2. Validate with `python validate_model.py`
3. Review `model_report.json` for detailed analysis
4. Deploy optimized model

## 📈 Future Enhancements

- **Weather Integration**: Real-time weather data
- **Soil Testing**: Integration with IoT sensors
- **Market Prices**: Live market price updates
- **Regional Varieties**: State-specific crop varieties
- **Disease Prediction**: Plant disease detection
- **Yield Optimization**: Advanced yield prediction

## 🔧 Troubleshooting

### Model Not Loading
```bash
# Regenerate model
python quick_train.py
```

### Low Accuracy
```bash
# Use comprehensive training
python train_model.py
```

### Missing Crops
- Add crop parameters to `crops_dataset.py`
- Retrain model with new crops

## 📞 Support

For Smart India Hackathon demo support:
- Check `validate_model.py` output for diagnostics
- Review `model_report.json` for model details
- Ensure all 31 crops are supported in predictions

---

**Built for Smart India Hackathon 2025** 🇮🇳
**KrishiMitra - Empowering Indian Farmers with AI** 🌾
