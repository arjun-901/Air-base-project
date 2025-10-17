#!/usr/bin/env python3
"""
Test script to verify the evaluation functionality works correctly.
"""

import requests
import pandas as pd
import numpy as np

def test_backend_evaluation():
    """Test the backend evaluation endpoint."""
    print("🧪 Testing Backend Evaluation Endpoint...")
    
    # Test data with some misclassifications
    test_data = {
        'true_aqi': [45.2, 78.3, 125.6, 156.7, 89.4, 200.5, 250.8, 320.1],
        'predicted_aqi': [47.8, 82.1, 118.9, 162.3, 91.2, 195.2, 245.1, 315.8],
        'model_name': 'ANN'
    }
    
    try:
        response = requests.post('http://localhost:8000/evaluate', json=test_data, timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ Backend evaluation endpoint working!")
            print(f"   Accuracy: {result['accuracy']:.3f}")
            print(f"   Precision: {result['precision']:.3f}")
            print(f"   Recall: {result['recall']:.3f}")
            print(f"   F1-Score: {result['f1_score']:.3f}")
            print(f"   Classes: {result['classes']}")
            print(f"   Confusion Matrix Shape: {len(result['confusion_matrix'])}x{len(result['confusion_matrix'][0])}")
            return True
        else:
            print(f"❌ Backend error: {response.status_code}")
            print(response.text)
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend not running. Start it with: uvicorn backend.app:app --reload --port 8000")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_aqi_categories():
    """Test the AQI categories endpoint."""
    print("\n🧪 Testing AQI Categories Endpoint...")
    
    try:
        response = requests.get('http://localhost:8000/aqi_categories', timeout=5)
        if response.status_code == 200:
            result = response.json()
            print("✅ AQI categories endpoint working!")
            print(f"   Number of categories: {len(result['categories'])}")
            for cat in result['categories']:
                print(f"   - {cat['name']}: {cat['range']}")
            return True
        else:
            print(f"❌ AQI categories error: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("⚠️ Backend not running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def create_test_evaluation_data():
    """Create test evaluation data file."""
    print("\n🧪 Creating Test Evaluation Data...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate realistic AQI data
    true_aqi = []
    
    # 40% Good (0-50)
    true_aqi.extend(np.random.uniform(0, 50, int(n_samples * 0.4)))
    
    # 30% Moderate (51-100)
    true_aqi.extend(np.random.uniform(51, 100, int(n_samples * 0.3)))
    
    # 15% Unhealthy for Sensitive Groups (101-150)
    true_aqi.extend(np.random.uniform(101, 150, int(n_samples * 0.15)))
    
    # 10% Unhealthy (151-200)
    true_aqi.extend(np.random.uniform(151, 200, int(n_samples * 0.1)))
    
    # 5% Very Unhealthy (201-300)
    true_aqi.extend(np.random.uniform(201, 300, int(n_samples * 0.05)))
    
    np.random.shuffle(true_aqi)
    
    # Add noise to predictions
    predicted_aqi = []
    for true_val in true_aqi:
        noise = np.random.normal(0, true_val * 0.1)  # 10% noise
        predicted_val = max(0, true_val + noise)
        predicted_aqi.append(predicted_val)
    
    # Create DataFrame
    eval_data = pd.DataFrame({
        'true_aqi': true_aqi,
        'predicted_aqi': predicted_aqi
    })
    
    # Save to CSV
    eval_data.to_csv('test_evaluation_comprehensive.csv', index=False)
    
    print(f"✅ Created test_evaluation_comprehensive.csv with {len(eval_data)} samples")
    print(f"   True AQI range: {eval_data['true_aqi'].min():.1f} - {eval_data['true_aqi'].max():.1f}")
    print(f"   Predicted AQI range: {eval_data['predicted_aqi'].min():.1f} - {eval_data['predicted_aqi'].max():.1f}")
    
    return True

def main():
    """Run all tests."""
    print("🚀 AirSense Evaluation System Test Suite")
    print("=" * 50)
    
    # Test backend
    backend_ok = test_backend_evaluation()
    categories_ok = test_aqi_categories()
    
    # Create test data
    data_ok = create_test_evaluation_data()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"   Backend Evaluation: {'✅ PASS' if backend_ok else '❌ FAIL'}")
    print(f"   AQI Categories: {'✅ PASS' if categories_ok else '❌ FAIL'}")
    print(f"   Test Data Creation: {'✅ PASS' if data_ok else '❌ FAIL'}")
    
    if backend_ok and categories_ok and data_ok:
        print("\n🎉 All tests passed! The evaluation system is ready to use.")
        print("\n📋 Next steps:")
        print("   1. Start the frontend: streamlit run frontend/streamlit_app.py")
        print("   2. Navigate to '📈 Model Evaluation' tab")
        print("   3. Upload test_evaluation_comprehensive.csv")
        print("   4. Select a model and click '🚀 Evaluate Model Performance'")
    else:
        print("\n⚠️ Some tests failed. Please check the backend and try again.")

if __name__ == "__main__":
    main()
