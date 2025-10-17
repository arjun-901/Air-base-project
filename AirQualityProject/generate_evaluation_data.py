#!/usr/bin/env python3
"""
Generate sample evaluation data for testing the confusion matrix and evaluation metrics functionality.
This script creates a CSV file with true and predicted AQI values for testing purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def classify_aqi(aqi_value: float) -> str:
    """Classify AQI value into categories based on standard AQI ranges."""
    if aqi_value <= 50:
        return "Good"
    elif aqi_value <= 100:
        return "Moderate"
    elif aqi_value <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200:
        return "Unhealthy"
    elif aqi_value <= 300:
        return "Very Unhealthy"
    else:
        return "Hazardous"

def generate_evaluation_data(n_samples=1000, noise_level=0.1):
    """
    Generate sample evaluation data with true and predicted AQI values.
    
    Args:
        n_samples: Number of samples to generate
        noise_level: Level of noise to add to predictions (0.0 to 1.0)
    """
    np.random.seed(42)
    
    # Generate true AQI values following a realistic distribution
    # Most values in moderate range, some in good, fewer in unhealthy ranges
    true_aqi = []
    
    # 40% Good (0-50)
    good_samples = int(n_samples * 0.4)
    true_aqi.extend(np.random.uniform(0, 50, good_samples))
    
    # 30% Moderate (51-100)
    moderate_samples = int(n_samples * 0.3)
    true_aqi.extend(np.random.uniform(51, 100, moderate_samples))
    
    # 15% Unhealthy for Sensitive Groups (101-150)
    usg_samples = int(n_samples * 0.15)
    true_aqi.extend(np.random.uniform(101, 150, usg_samples))
    
    # 10% Unhealthy (151-200)
    unhealthy_samples = int(n_samples * 0.1)
    true_aqi.extend(np.random.uniform(151, 200, unhealthy_samples))
    
    # 5% Very Unhealthy (201-300)
    very_unhealthy_samples = int(n_samples * 0.05)
    true_aqi.extend(np.random.uniform(201, 300, very_unhealthy_samples))
    
    # Shuffle the data
    np.random.shuffle(true_aqi)
    
    # Generate predicted AQI values with some noise
    predicted_aqi = []
    for true_val in true_aqi:
        # Add noise proportional to the true value
        noise = np.random.normal(0, true_val * noise_level)
        predicted_val = max(0, true_val + noise)  # Ensure non-negative
        predicted_aqi.append(predicted_val)
    
    # Create DataFrame
    eval_data = pd.DataFrame({
        'true_aqi': true_aqi,
        'predicted_aqi': predicted_aqi
    })
    
    # Add some categorical columns for reference
    eval_data['true_category'] = eval_data['true_aqi'].apply(classify_aqi)
    eval_data['predicted_category'] = eval_data['predicted_aqi'].apply(classify_aqi)
    
    return eval_data

def main():
    """Generate and save evaluation data."""
    print("Generating sample evaluation data...")
    
    # Generate data with different noise levels
    eval_data_low_noise = generate_evaluation_data(n_samples=1000, noise_level=0.05)
    eval_data_medium_noise = generate_evaluation_data(n_samples=1000, noise_level=0.15)
    eval_data_high_noise = generate_evaluation_data(n_samples=1000, noise_level=0.3)
    
    # Save to CSV files
    eval_data_low_noise.to_csv('evaluation_data_low_noise.csv', index=False)
    eval_data_medium_noise.to_csv('evaluation_data_medium_noise.csv', index=False)
    eval_data_high_noise.to_csv('evaluation_data_high_noise.csv', index=False)
    
    print("✅ Generated evaluation data files:")
    print("  - evaluation_data_low_noise.csv (5% noise)")
    print("  - evaluation_data_medium_noise.csv (15% noise)")
    print("  - evaluation_data_high_noise.csv (30% noise)")
    
    # Show sample statistics
    print("\n📊 Sample statistics (low noise):")
    print(f"  True AQI range: {eval_data_low_noise['true_aqi'].min():.1f} - {eval_data_low_noise['true_aqi'].max():.1f}")
    print(f"  Predicted AQI range: {eval_data_low_noise['predicted_aqi'].min():.1f} - {eval_data_low_noise['predicted_aqi'].max():.1f}")
    
    print("\n📋 Category distribution (true values):")
    category_counts = eval_data_low_noise['true_category'].value_counts()
    for category, count in category_counts.items():
        print(f"  {category}: {count} samples ({count/len(eval_data_low_noise)*100:.1f}%)")
    
    print("\n🎯 Use these files in the Model Evaluation section of the AirSense dashboard!")

if __name__ == "__main__":
    main()
