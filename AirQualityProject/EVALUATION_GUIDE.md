# Model Evaluation & Confusion Matrix Guide

## Overview
The AirSense platform now includes comprehensive model evaluation capabilities with confusion matrix analysis, precision, recall, F1-score, and other standard evaluation metrics.

## Features Added

### Backend (`backend/app.py`)
- **AQI Classification**: Automatically classifies AQI values into 6 categories:
  - Good (0-50)
  - Moderate (51-100) 
  - Unhealthy for Sensitive Groups (101-150)
  - Unhealthy (151-200)
  - Very Unhealthy (201-300)
  - Hazardous (301+)

- **Evaluation Metrics**: Calculates comprehensive metrics:
  - Accuracy = (TP + TN) / (TP + TN + FP + FN)
  - Precision = TP / (TP + FP)
  - Recall (Sensitivity) = TP / (TP + FN)
  - Specificity (TNR) = TN / (TN + FP)
  - F1 Score = 2 × (Precision × Recall) / (Precision + Recall)

- **New API Endpoints**:
  - `POST /evaluate` - Evaluate model performance
  - `GET /aqi_categories` - Get AQI category definitions

### Frontend (`frontend/streamlit_app.py`)
- **New Navigation Tab**: "📈 Model Evaluation"
- **Confusion Matrix Visualization**: Interactive heatmap using Plotly
- **Comprehensive Metrics Display**: All evaluation metrics in organized layout
- **Per-Class Performance**: Detailed breakdown by AQI category
- **Performance Insights**: Automated analysis and recommendations

## How to Use

### 1. Prepare Evaluation Data
Create a CSV file with two columns:
- `true_aqi`: Actual AQI values
- `predicted_aqi`: Model-predicted AQI values

### 2. Sample Data Generation
Run the included data generator:
```bash
cd /home/arjun/Desktop/Air-base-project/AirQualityProject
source venv/bin/activate
python generate_evaluation_data.py
```

This creates three sample files:
- `evaluation_data_low_noise.csv` (5% noise)
- `evaluation_data_medium_noise.csv` (15% noise) 
- `evaluation_data_high_noise.csv` (30% noise)

### 3. Using the Evaluation Interface

1. **Start the Backend**:
   ```bash
   cd /home/arjun/Desktop/Air-base-project/AirQualityProject
   source venv/bin/activate
   uvicorn backend.app:app --reload --port 8000
   ```

2. **Start the Frontend**:
   ```bash
   cd /home/arjun/Desktop/Air-base-project/AirQualityProject
   source venv/bin/activate
   streamlit run frontend/streamlit_app.py --server.port 8501
   ```

3. **Navigate to Evaluation**:
   - Open http://localhost:8501
   - Click "📈 Model Evaluation" tab
   - Upload your evaluation CSV file
   - Select the model name
   - Click "🚀 Evaluate Model Performance"

### 4. Understanding the Results

#### Confusion Matrix
- **Rows**: True AQI categories
- **Columns**: Predicted AQI categories
- **Diagonal**: Correct predictions
- **Off-diagonal**: Misclassifications

#### Key Metrics
- **Accuracy**: Overall correctness percentage
- **Precision**: How many predicted positives were actually positive
- **Recall**: How many actual positives were correctly identified
- **F1-Score**: Harmonic mean of precision and recall

#### Performance Insights
- **Best Category**: Which AQI level the model predicts most accurately
- **Needs Improvement**: Which category needs better training
- **Overall Assessment**: Performance level recommendations

## Example Evaluation Data Format

```csv
true_aqi,predicted_aqi
45.2,47.8
78.3,82.1
125.6,118.9
156.7,162.3
89.4,91.2
```

## Troubleshooting

### Common Issues
1. **Missing Columns**: Ensure CSV has `true_aqi` and `predicted_aqi` columns
2. **Invalid Values**: AQI values should be non-negative numbers
3. **Backend Connection**: Ensure backend is running on port 8000
4. **Dependencies**: Install required packages with `pip install -r requirements.txt`

### Performance Interpretation
- **Accuracy > 80%**: Excellent performance
- **Accuracy 60-80%**: Good performance, consider fine-tuning
- **Accuracy < 60%**: Needs improvement, consider retraining

## Technical Details

### AQI Classification Logic
```python
def classify_aqi(aqi_value: float) -> str:
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
```

### Evaluation Metrics Calculation
All metrics are calculated using scikit-learn's built-in functions:
- `confusion_matrix()` for confusion matrix
- `accuracy_score()` for accuracy
- `precision_score()` for precision
- `recall_score()` for recall
- `f1_score()` for F1-score

## Files Modified/Created
- `backend/app.py` - Added evaluation endpoints and metrics calculation
- `frontend/streamlit_app.py` - Added evaluation interface
- `requirements.txt` - Added new dependencies
- `generate_evaluation_data.py` - Sample data generator
- `EVALUATION_GUIDE.md` - This documentation

## Dependencies Added
- `seaborn` - For enhanced plotting
- `plotly` - For interactive visualizations
- `pydantic` - For data validation
- `scikit-learn` - For evaluation metrics (already present)
