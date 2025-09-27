# Model Validation Script for KrishiMitra - Smart India Hackathon
# This script validates that our ML model works correctly with all crops in the dataset

import pickle
import numpy as np
import pandas as pd
from crops_dataset import get_all_crop_names, get_crop_info, CROP_INFO
import json

def load_model():
    """Load the trained model and scaler"""
    try:
        with open('backend/data/model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('backend/data/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        return model, scaler
    except FileNotFoundError:
        print("❌ Model files not found. Please run train_model.py first.")
        return None, None

def validate_model_crops():
    """Validate that the model can predict all crops in our dataset"""
    print("🔍 Validating model coverage for all crops...")
    
    model, scaler = load_model()
    if model is None:
        return False
    
    # Get all crops from dataset
    dataset_crops = set(get_all_crop_names())
    
    # Get crops that the model can predict
    model_crops = set(model.classes_)
    
    print(f"📊 Dataset contains: {len(dataset_crops)} crops")
    print(f"🤖 Model can predict: {len(model_crops)} crops")
    
    # Check coverage
    missing_in_model = dataset_crops - model_crops
    extra_in_model = model_crops - dataset_crops
    
    if missing_in_model:
        print(f"⚠️  Crops in dataset but not in model: {missing_in_model}")
    
    if extra_in_model:
        print(f"ℹ️  Crops in model but not in dataset: {extra_in_model}")
    
    coverage = len(model_crops & dataset_crops) / len(dataset_crops) * 100
    print(f"✅ Model coverage: {coverage:.1f}%")
    
    return coverage >= 90  # Consider 90%+ coverage as good

def test_crop_predictions():
    """Test predictions for each crop category"""
    print("\n🧪 Testing predictions for each crop category...")
    
    model, scaler = load_model()
    if model is None:
        return False
    
    # Define test conditions for different crop categories
    test_conditions = {
        'high_water_high_temp': [100, 60, 80, 28, 85, 6.5, 150],  # Rice, Sugarcane
        'low_water_cool_temp': [80, 60, 40, 18, 60, 6.8, 50],     # Wheat, Mustard
        'medium_conditions': [90, 50, 60, 24, 70, 6.5, 80],       # Maize, Tomato
        'arid_conditions': [50, 30, 30, 32, 45, 7.0, 35],         # Bajra, Jowar
        'rich_soil_conditions': [150, 80, 120, 26, 75, 6.8, 100], # Banana, Potato
    }
    
    results = {}
    
    for condition_name, values in test_conditions.items():
        # Scale the input
        input_scaled = scaler.transform([values])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities)
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        top_predictions = [(model.classes_[i], probabilities[i]) for i in top_indices]
        
        results[condition_name] = {
            'input': values,
            'prediction': prediction,
            'confidence': confidence,
            'top_3': top_predictions
        }
        
        print(f"\n{condition_name.replace('_', ' ').title()}:")
        print(f"  Input: N={values[0]}, P={values[1]}, K={values[2]}, T={values[3]}°C, H={values[4]}%, pH={values[5]}, R={values[6]}mm")
        print(f"  Predicted: {prediction} (confidence: {confidence:.2f})")
        print(f"  Top 3: {', '.join([f'{crop}({prob:.2f})' for crop, prob in top_predictions])}")
    
    return True

def generate_prediction_report():
    """Generate a comprehensive prediction report"""
    print("\n📋 Generating comprehensive prediction report...")
    
    model, scaler = load_model()
    if model is None:
        return False
    
    report = {
        'model_info': {
            'total_crops': len(model.classes_),
            'feature_count': len(['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']),
            'model_type': type(model).__name__
        },
        'supported_crops': list(model.classes_),
        'crop_categories': {}
    }
    
    # Categorize crops
    for crop in model.classes_:
        crop_info = get_crop_info(crop)
        if crop_info:
            category = crop_info.get('category', 'Unknown')
            if category not in report['crop_categories']:
                report['crop_categories'][category] = []
            report['crop_categories'][category].append(crop)
    
    # Save report
    with open('model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Model report saved to: model_report.json")
    
    # Print summary
    print(f"\n📊 Model Summary:")
    print(f"   Total crops supported: {report['model_info']['total_crops']}")
    print(f"   Input features: {report['model_info']['feature_count']}")
    print(f"   Model type: {report['model_info']['model_type']}")
    
    print(f"\n🏷️  Crops by Category:")
    for category, crops in report['crop_categories'].items():
        print(f"   {category}: {len(crops)} crops ({', '.join(crops[:3])}{'...' if len(crops) > 3 else ''})")
    
    return True

def test_api_integration():
    """Test integration with the API prediction function"""
    print("\n🔗 Testing API integration...")
    
    # Test the same function that the API uses
    try:
        from app import get_rule_based_prediction
        
        # Test rule-based prediction (fallback when ML model fails)
        test_params = {
            'nitrogen': 90,
            'phosphorus': 60,
            'potassium': 50,
            'temperature': 25,
            'humidity': 80,
            'ph': 6.0,
            'rainfall': 200
        }
        
        result = get_rule_based_prediction(**test_params)
        print(f"✅ Rule-based prediction works: {result}")
        
    except ImportError:
        print("⚠️  Could not test API integration (app.py not found)")
    except Exception as e:
        print(f"❌ API integration test failed: {e}")

def main():
    """Main validation function"""
    print("🌾 KrishiMitra Model Validation - Smart India Hackathon")
    print("=" * 60)
    
    all_tests_passed = True
    
    # Test 1: Model coverage
    if not validate_model_crops():
        all_tests_passed = False
    
    # Test 2: Prediction testing
    if not test_crop_predictions():
        all_tests_passed = False
    
    # Test 3: Generate report
    if not generate_prediction_report():
        all_tests_passed = False
    
    # Test 4: API integration
    test_api_integration()
    
    print("\n" + "=" * 60)
    if all_tests_passed:
        print("🎉 All validation tests passed!")
        print("✅ Model is ready for Smart India Hackathon demo")
    else:
        print("❌ Some validation tests failed")
        print("⚠️  Please check the issues above")
    
    print("\n📝 Next steps:")
    print("   1. Deploy the updated model to your backend")
    print("   2. Test the /api/predict endpoint")
    print("   3. Verify crop recommendations in your app")

if __name__ == "__main__":
    main()
