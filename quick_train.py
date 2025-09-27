# Quick Training Script for KrishiMitra - Smart India Hackathon
# This is a simplified version for quick model generation

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

def quick_generate_data():
    """Generate training data quickly for all crops"""
    
    # All crops with their optimal growing conditions
    crops_data = {
        # Format: [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        'rice': [100, 60, 50, 27, 85, 6.0, 200],
        'wheat': [125, 80, 60, 20, 60, 6.5, 75],
        'maize': [110, 60, 60, 24, 70, 6.5, 90],
        'bajra': [60, 30, 30, 30, 50, 7.0, 45],
        'jowar': [80, 45, 45, 28, 55, 7.5, 55],
        'ragi': [70, 35, 35, 25, 70, 6.0, 80],
        'chickpea': [30, 60, 30, 25, 60, 7.0, 40],
        'lentil': [25, 50, 25, 24, 60, 7.0, 35],
        'moong': [30, 45, 30, 30, 70, 7.0, 42],
        'sugarcane': [200, 90, 115, 25, 80, 6.5, 200],
        'cotton': [110, 60, 60, 25, 65, 7.0, 75],
        'tobacco': [90, 60, 90, 25, 70, 6.5, 55],
        'jute': [80, 45, 45, 30, 87, 7.0, 150],
        'mustard': [80, 60, 30, 20, 60, 7.0, 40],
        'groundnut': [30, 60, 60, 25, 70, 6.5, 75],
        'soybean': [45, 90, 60, 25, 70, 6.5, 80],
        'tea': [150, 75, 75, 25, 80, 5.5, 160],
        'coffee': [115, 60, 90, 20, 77, 6.5, 140],
        'ginger': [125, 75, 125, 27, 87, 6.5, 160],
        'saffron': [60, 30, 60, 17, 60, 7.0, 40],
        'mango': [115, 60, 115, 27, 70, 6.5, 112],
        'banana': [200, 75, 250, 28, 80, 7.0, 160],
        'tomato': [100, 60, 100, 22, 70, 6.5, 65],
        'potato': [125, 75, 125, 17, 75, 6.0, 65],
        'onion': [100, 60, 100, 20, 65, 7.0, 47],
        'brinjal': [125, 75, 125, 27, 70, 6.5, 80],
        'cabbage': [150, 90, 150, 17, 75, 6.5, 50],
        'cauliflower': [150, 90, 150, 17, 75, 6.5, 50],
        'peas': [30, 90, 60, 14, 70, 7.0, 40],
        'carrot': [100, 60, 100, 18, 70, 6.5, 50],
        'capsicum': [125, 75, 125, 22, 70, 6.5, 80]
    }
    
    # Generate multiple samples for each crop with variations
    training_data = []
    samples_per_crop = 100
    
    for crop, base_values in crops_data.items():
        for _ in range(samples_per_crop):
            # Add random variations (±15% of base value)
            sample = []
            for i, base_val in enumerate(base_values):
                variation = np.random.uniform(-0.15, 0.15) * base_val
                new_val = base_val + variation
                
                # Ensure realistic bounds
                if i == 3:  # temperature
                    new_val = max(5, min(45, new_val))
                elif i == 4:  # humidity
                    new_val = max(30, min(95, new_val))
                elif i == 5:  # pH
                    new_val = max(4.0, min(9.0, new_val))
                else:
                    new_val = max(0, new_val)
                
                sample.append(new_val)
            
            sample.append(crop)
            training_data.append(sample)
    
    # Create DataFrame
    columns = ['nitrogen', 'phosphorus', 'potassium', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
    df = pd.DataFrame(training_data, columns=columns)
    
    return df

def train_quick_model():
    """Train a model quickly"""
    print("🚀 Quick training model for all crops...")
    
    # Generate data
    df = quick_generate_data()
    print(f"Generated {len(df)} samples for {len(df['label'].unique())} crops")
    
    # Prepare data
    X = df.drop('label', axis=1)
    y = df['label']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    # Calculate training accuracy
    train_accuracy = model.score(X_scaled, y)
    print(f"Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # Create backend/data directory if it doesn't exist
    os.makedirs('backend/data', exist_ok=True)
    
    # Save model and scaler
    with open('backend/data/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('backend/data/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("✅ Model and scaler saved successfully!")
    print(f"📊 Supports {len(model.classes_)} crops: {', '.join(sorted(model.classes_))}")
    
    return model, scaler

def test_predictions():
    """Test some predictions"""
    print("\n🧪 Testing predictions...")
    
    # Load model
    with open('backend/data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('backend/data/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Test cases
    test_cases = [
        ([100, 60, 50, 27, 85, 6.0, 200], "rice"),
        ([125, 80, 60, 20, 60, 6.5, 75], "wheat"),
        ([100, 60, 100, 22, 70, 6.5, 65], "tomato"),
        ([110, 60, 60, 25, 65, 7.0, 75], "cotton"),
        ([125, 75, 125, 17, 75, 6.0, 65], "potato")
    ]
    
    for inputs, expected in test_cases:
        input_scaled = scaler.transform([inputs])
        prediction = model.predict(input_scaled)[0]
        confidence = max(model.predict_proba(input_scaled)[0])
        
        status = "✅" if prediction == expected else "❌"
        print(f"{status} Expected: {expected}, Got: {prediction} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    print("🌾 KrishiMitra Quick Model Training")
    print("=" * 50)
    
    model, scaler = train_quick_model()
    test_predictions()
    
    print("\n🎉 Quick training completed!")
    print("💡 For better accuracy, run 'python train_model.py' for comprehensive training")
