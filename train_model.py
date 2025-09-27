# Comprehensive ML Model Training for KrishiMitra - Smart India Hackathon
# This script creates training data and trains a model for all crops in our dataset

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os
from crops_dataset import get_all_crop_names, get_crop_info

# Set random seed for reproducibility
np.random.seed(42)

def generate_training_data():
    """Generate synthetic training data for all crops in our dataset"""
    
    # Get all crops from our dataset
    all_crops = get_all_crop_names()
    print(f"Generating training data for {len(all_crops)} crops...")
    
    # Training data will be stored here
    training_data = []
    
    # Define crop-specific parameter ranges based on agricultural knowledge
    crop_parameters = {
        # Cereals
        'rice': {
            'nitrogen': (80, 120), 'phosphorus': (40, 80), 'potassium': (40, 80),
            'temperature': (20, 35), 'humidity': (70, 90), 'ph': (5.5, 7.0), 'rainfall': (150, 300)
        },
        'wheat': {
            'nitrogen': (100, 150), 'phosphorus': (60, 100), 'potassium': (40, 80),
            'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (50, 100)
        },
        'maize': {
            'nitrogen': (80, 140), 'phosphorus': (40, 80), 'potassium': (40, 80),
            'temperature': (21, 27), 'humidity': (60, 80), 'ph': (5.8, 7.0), 'rainfall': (60, 120)
        },
        'bajra': {
            'nitrogen': (40, 80), 'phosphorus': (20, 40), 'potassium': (20, 40),
            'temperature': (25, 35), 'humidity': (40, 60), 'ph': (6.0, 8.0), 'rainfall': (30, 60)
        },
        'jowar': {
            'nitrogen': (60, 100), 'phosphorus': (30, 60), 'potassium': (30, 60),
            'temperature': (26, 30), 'humidity': (45, 65), 'ph': (6.0, 8.5), 'rainfall': (40, 75)
        },
        'ragi': {
            'nitrogen': (50, 90), 'phosphorus': (25, 50), 'potassium': (25, 50),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.0, 7.0), 'rainfall': (60, 100)
        },
        
        # Pulses
        'chickpea': {
            'nitrogen': (20, 40), 'phosphorus': (40, 80), 'potassium': (20, 40),
            'temperature': (20, 30), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (30, 50)
        },
        'lentil': {
            'nitrogen': (15, 35), 'phosphorus': (35, 70), 'potassium': (15, 35),
            'temperature': (18, 30), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (25, 45)
        },
        'moong': {
            'nitrogen': (20, 40), 'phosphorus': (30, 60), 'potassium': (20, 40),
            'temperature': (25, 35), 'humidity': (60, 80), 'ph': (6.5, 7.5), 'rainfall': (35, 50)
        },
        
        # Cash Crops
        'sugarcane': {
            'nitrogen': (150, 250), 'phosphorus': (60, 120), 'potassium': (80, 150),
            'temperature': (20, 30), 'humidity': (70, 90), 'ph': (6.0, 7.5), 'rainfall': (150, 250)
        },
        'cotton': {
            'nitrogen': (80, 140), 'phosphorus': (40, 80), 'potassium': (40, 80),
            'temperature': (21, 30), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 100)
        },
        'tobacco': {
            'nitrogen': (60, 120), 'phosphorus': (40, 80), 'potassium': (60, 120),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (5.5, 7.0), 'rainfall': (40, 75)
        },
        'jute': {
            'nitrogen': (60, 100), 'phosphorus': (30, 60), 'potassium': (30, 60),
            'temperature': (25, 35), 'humidity': (80, 95), 'ph': (6.0, 7.5), 'rainfall': (120, 180)
        },
        
        # Oilseeds
        'mustard': {
            'nitrogen': (60, 100), 'phosphorus': (40, 80), 'potassium': (20, 40),
            'temperature': (15, 25), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (30, 50)
        },
        'groundnut': {
            'nitrogen': (20, 40), 'phosphorus': (40, 80), 'potassium': (40, 80),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (50, 100)
        },
        'soybean': {
            'nitrogen': (30, 60), 'phosphorus': (60, 120), 'potassium': (40, 80),
            'temperature': (20, 30), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (60, 100)
        },
        
        # Plantation
        'tea': {
            'nitrogen': (100, 200), 'phosphorus': (50, 100), 'potassium': (50, 100),
            'temperature': (20, 30), 'humidity': (70, 90), 'ph': (4.5, 6.0), 'rainfall': (120, 200)
        },
        'coffee': {
            'nitrogen': (80, 150), 'phosphorus': (40, 80), 'potassium': (60, 120),
            'temperature': (15, 25), 'humidity': (70, 85), 'ph': (6.0, 7.0), 'rainfall': (100, 180)
        },
        
        # Spices
        'ginger': {
            'nitrogen': (100, 150), 'phosphorus': (50, 100), 'potassium': (100, 150),
            'temperature': (25, 30), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (120, 200)
        },
        'saffron': {
            'nitrogen': (40, 80), 'phosphorus': (20, 40), 'potassium': (40, 80),
            'temperature': (15, 20), 'humidity': (50, 70), 'ph': (6.0, 7.5), 'rainfall': (30, 50)
        },
        
        # Fruits
        'mango': {
            'nitrogen': (80, 150), 'phosphorus': (40, 80), 'potassium': (80, 150),
            'temperature': (24, 30), 'humidity': (60, 80), 'ph': (5.5, 7.5), 'rainfall': (75, 150)
        },
        'banana': {
            'nitrogen': (150, 250), 'phosphorus': (50, 100), 'potassium': (200, 300),
            'temperature': (26, 30), 'humidity': (75, 85), 'ph': (6.0, 7.5), 'rainfall': (120, 200)
        },
        
        # Vegetables
        'tomato': {
            'nitrogen': (80, 120), 'phosphorus': (40, 80), 'potassium': (80, 120),
            'temperature': (20, 25), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (50, 80)
        },
        'potato': {
            'nitrogen': (100, 150), 'phosphorus': (50, 100), 'potassium': (100, 150),
            'temperature': (15, 20), 'humidity': (70, 80), 'ph': (5.0, 6.5), 'rainfall': (50, 80)
        },
        'onion': {
            'nitrogen': (80, 120), 'phosphorus': (40, 80), 'potassium': (80, 120),
            'temperature': (15, 25), 'humidity': (60, 70), 'ph': (6.0, 7.5), 'rainfall': (35, 60)
        },
        'brinjal': {
            'nitrogen': (100, 150), 'phosphorus': (50, 100), 'potassium': (100, 150),
            'temperature': (22, 32), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (60, 100)
        },
        'cabbage': {
            'nitrogen': (120, 180), 'phosphorus': (60, 120), 'potassium': (120, 180),
            'temperature': (15, 20), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (40, 60)
        },
        'cauliflower': {
            'nitrogen': (120, 180), 'phosphorus': (60, 120), 'potassium': (120, 180),
            'temperature': (15, 20), 'humidity': (70, 80), 'ph': (6.0, 7.0), 'rainfall': (40, 60)
        },
        'peas': {
            'nitrogen': (20, 40), 'phosphorus': (60, 120), 'potassium': (40, 80),
            'temperature': (10, 18), 'humidity': (60, 80), 'ph': (6.0, 7.5), 'rainfall': (30, 50)
        },
        'carrot': {
            'nitrogen': (80, 120), 'phosphorus': (40, 80), 'potassium': (80, 120),
            'temperature': (16, 20), 'humidity': (65, 75), 'ph': (6.0, 7.0), 'rainfall': (40, 60)
        },
        'capsicum': {
            'nitrogen': (100, 150), 'phosphorus': (50, 100), 'potassium': (100, 150),
            'temperature': (20, 25), 'humidity': (60, 80), 'ph': (6.0, 7.0), 'rainfall': (60, 100)
        }
    }
    
    # Generate samples for each crop
    samples_per_crop = 150  # Generate 150 samples per crop
    
    for crop in all_crops:
        if crop in crop_parameters:
            params = crop_parameters[crop]
            
            for _ in range(samples_per_crop):
                # Generate random values within the specified ranges for each parameter
                sample = []
                for param in ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall']:
                    min_val, max_val = params[param]
                    # Add some noise to make data more realistic
                    value = np.random.uniform(min_val, max_val)
                    # Add small random variations
                    noise = np.random.normal(0, (max_val - min_val) * 0.05)
                    value = max(0, value + noise)  # Ensure non-negative values
                    sample.append(value)
                
                sample.append(crop)  # Add crop label
                training_data.append(sample)
        else:
            print(f"Warning: No parameters defined for crop '{crop}', skipping...")
    
    # Create DataFrame
    columns = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    df = pd.DataFrame(training_data, columns=columns)
    
    print(f"Generated {len(df)} training samples for {len(df['label'].unique())} crops")
    print("Crop distribution:")
    print(df['label'].value_counts())
    
    return df

def train_model(df):
    """Train the machine learning model"""
    print("\nTraining machine learning model...")
    
    # Separate features and labels
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_names = X.columns
    importance_scores = model.feature_importances_
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance_scores
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model, scaler, accuracy

def save_model(model, scaler):
    """Save the trained model and scaler"""
    # Create backend/data directory if it doesn't exist
    os.makedirs('backend/data', exist_ok=True)
    
    # Save model
    with open('backend/data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Save scaler
    with open('backend/data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\nModel and scaler saved successfully!")
    print("- Model saved to: backend/data/model.pkl")
    print("- Scaler saved to: backend/data/scaler.pkl")

def test_model_predictions():
    """Test the model with some sample predictions"""
    print("\nTesting model with sample predictions...")
    
    # Load the saved model and scaler
    with open('backend/data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('backend/data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Test cases for different crops
    test_cases = [
        {
            'name': 'Rice conditions',
            'data': [90, 60, 50, 25, 80, 6.0, 200],
            'expected': 'rice'
        },
        {
            'name': 'Wheat conditions', 
            'data': [120, 80, 60, 20, 60, 6.5, 75],
            'expected': 'wheat'
        },
        {
            'name': 'Tomato conditions',
            'data': [100, 60, 100, 22, 70, 6.5, 65],
            'expected': 'tomato'
        },
        {
            'name': 'Cotton conditions',
            'data': [110, 60, 60, 25, 65, 7.0, 75],
            'expected': 'cotton'
        }
    ]
    
    for test_case in test_cases:
        # Scale the input
        input_scaled = scaler.transform([test_case['data']])
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = max(probabilities)
        
        print(f"\n{test_case['name']}:")
        print(f"  Input: N={test_case['data'][0]}, P={test_case['data'][1]}, K={test_case['data'][2]}, "
              f"T={test_case['data'][3]}°C, H={test_case['data'][4]}%, pH={test_case['data'][5]}, "
              f"R={test_case['data'][6]}mm")
        print(f"  Predicted: {prediction} (confidence: {confidence:.2f})")
        print(f"  Expected: {test_case['expected']}")
        print(f"  ✅ Correct" if prediction == test_case['expected'] else f"  ❌ Incorrect")

if __name__ == "__main__":
    print("🌾 KrishiMitra ML Model Training - Smart India Hackathon")
    print("=" * 60)
    
    # Generate training data
    df = generate_training_data()
    
    # Save training data for reference
    df.to_csv('training_data.csv', index=False)
    print(f"\nTraining data saved to: training_data.csv")
    
    # Train the model
    model, scaler, accuracy = train_model(df)
    
    # Save the model
    save_model(model, scaler)
    
    # Test the model
    test_model_predictions()
    
    print(f"\n🎉 Model training completed successfully!")
    print(f"📊 Final Accuracy: {accuracy*100:.2f}%")
    print(f"🌱 Supports {len(df['label'].unique())} different crops")
    print(f"📁 Model files saved in backend/data/")
