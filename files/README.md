# Restaurant Review Sentiment Analysis - Full Stack NLP Application

A complete Natural Language Processing system for sentiment analysis of restaurant reviews, featuring a Flask backend API and a beautiful React frontend dashboard.

## 🎯 Features

### Backend (Flask API)
- ✅ Multiple ML model training (Decision Tree, Random Forest, Logistic Regression, Naive Bayes, KNN, SVM)
- ✅ TF-IDF vectorization for feature extraction
- ✅ Text preprocessing (stopword removal, stemming)
- ✅ REST API endpoints for predictions and model management
- ✅ Model persistence (save/load trained models)
- ✅ Comprehensive metrics (accuracy, precision, recall, F1, confusion matrix)

### Frontend (React Dashboard)
- ✅ Real-time sentiment prediction
- ✅ Model performance comparison
- ✅ Interactive confusion matrix visualization
- ✅ Performance metrics dashboard
- ✅ Beautiful cyberpunk UI with animations
- ✅ Demo mode (works without backend)
- ✅ Responsive design

## 📁 Project Structure

```
nlp-sentiment-analysis/
├── backend_api.py              # Flask REST API
├── requirements.txt            # Python dependencies
├── frontend_with_api.html      # React dashboard (API-integrated)
├── nlp-dashboard.html          # React dashboard (standalone demo)
├── Restaurant_Reviews.tsv      # Dataset (you need to provide this)
├── model.pkl                   # Trained model (generated after training)
└── vectorizer.pkl              # TF-IDF vectorizer (generated after training)
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Dependencies:**
- Flask==3.0.0
- flask-cors==4.0.0
- numpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- nltk==3.8.1

### Step 2: Prepare Your Dataset

Place your `Restaurant_Reviews.tsv` file in the same directory as `backend_api.py`.

**Dataset Format:**
- Tab-separated values (.tsv)
- Column 1: Review (text)
- Column 2: Liked (0 or 1)

Example:
```
Review	Liked
Wow... Loved this place.	1
Crust is not good.	0
Not tasty and the texture was just nasty.	0
```

### Step 3: Start the Backend

```bash
python backend_api.py
```

The API will start on `http://localhost:5000`

You should see:
```
🚀 Restaurant Review Sentiment Analysis API
============================================================

Available Endpoints:
  GET  /api/health          - Health check
  POST /api/train           - Train models
  POST /api/predict         - Predict sentiment
  GET  /api/models          - Get model comparison
  GET  /api/metrics         - Get performance metrics
  GET  /api/dataset-info    - Get dataset information
  POST /api/preprocess      - Preprocess text

============================================================

 * Running on http://0.0.0.0:5000
```

### Step 4: Open the Frontend

Open `frontend_with_api.html` in your web browser.

### Step 5: Train the Models

1. Click on the **"Models"** tab
2. Click **"Train Models"** button
3. Wait for training to complete (30-60 seconds)
4. Models are now ready for predictions!

## 📊 API Usage

### Health Check
```bash
curl http://localhost:5000/api/health
```

### Train Models
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json"
```

### Predict Sentiment
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The food was amazing and service was excellent!",
    "model": "Decision Tree"
  }'
```

Response:
```json
{
  "sentiment": "Positive",
  "prediction": 1,
  "confidence": 0.89,
  "model_used": "Decision Tree"
}
```

### Get Model Comparison
```bash
curl http://localhost:5000/api/models
```

### Get Performance Metrics
```bash
curl http://localhost:5000/api/metrics
```

## 🎨 Frontend Features

### Predict Tab
- Enter restaurant reviews
- Select ML model
- Get instant sentiment predictions
- View confidence scores

### Models Tab
- Compare 6+ ML algorithms
- Visual performance bars
- See accuracy rankings
- Train new models

### Metrics Tab
- Confusion matrix visualization
- Precision, Recall, F1 Score
- Specificity metrics
- Model performance stats

### Setup Tab
- Backend connection status
- Setup instructions
- API endpoint documentation

## 🧠 Supported Models

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| XGBoost* | Boosting | ~85% |
| LGBM* | Boosting | ~84% |
| Random Forest | Ensemble | ~82% |
| SVM | Kernel | ~80% |
| Logistic Regression | Linear | ~78% |
| Naive Bayes | Probabilistic | ~77% |
| KNN | Instance | ~75% |
| Decision Tree | Baseline | ~73% |

*Note: XGBoost and LGBM require additional installation:
```bash
pip install xgboost lightgbm
```

## 🔧 Advanced Configuration

### Change Dataset Path

```python
# In backend_api.py or via API
analyzer = SentimentAnalyzer(data_path='path/to/your/data.tsv')
```

### Adjust Train/Test Split

```python
# In backend_api.py, train_models method
results, X_train, X_test, y_train, y_test = analyzer.train_models(
    X, y, 
    test_size=0.20  # Change to 0.30 for 30% test split
)
```

### Add More Models

Edit the `train_models` method in `backend_api.py`:

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

models = {
    'Decision Tree': DecisionTreeClassifier(),
    'XGBoost': XGBClassifier(),
    'LGBM': LGBMClassifier(),
    # ... add more models
}
```

## 📈 Model Performance Tips

### Increase Accuracy
1. **Add more training data** (>2000 reviews recommended)
2. **Try ensemble methods** (XGBoost, LGBM, Random Forest)
3. **Experiment with n-grams**:
   ```python
   vectorizer = TfidfVectorizer(max_features=1500, ngram_range=(1,2))
   ```
4. **Use Word2Vec or BERT embeddings**
5. **Implement hyperparameter tuning**:
   ```python
   from sklearn.model_selection import GridSearchCV
   ```

### Reduce Overfitting
- Increase test size
- Use cross-validation
- Add regularization
- Reduce max_features in TF-IDF

## 🐛 Troubleshooting

### "NLTK stopwords not found"
```python
import nltk
nltk.download('stopwords')
```

### "CORS Error"
Make sure flask-cors is installed and the frontend is accessing the correct API URL.

### "Model not trained"
Click "Train Models" in the Models tab or call `/api/train` endpoint.

### "Dataset not found"
Ensure `Restaurant_Reviews.tsv` is in the same directory as `backend_api.py`.

### Port 5000 already in use
```bash
# Change port in backend_api.py
app.run(debug=True, host='0.0.0.0', port=5001)

# Update API_BASE_URL in frontend_with_api.html
const API_BASE_URL = 'http://localhost:5001/api';
```

## 📝 Code Examples

### Custom Prediction Script

```python
from backend_api import SentimentAnalyzer

# Initialize
analyzer = SentimentAnalyzer()

# Load saved model
analyzer.load_model()

# Predict
result = analyzer.predict_sentiment(
    "The food was delicious but service was slow",
    model_name="Random Forest"
)

print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Batch Predictions

```python
reviews = [
    "Amazing experience!",
    "Terrible food and service",
    "It was okay, nothing special"
]

for review in reviews:
    result = analyzer.predict_sentiment(review)
    print(f"{review} → {result['sentiment']}")
```

## 🎓 Learning Resources

### NLP Preprocessing
- Tokenization: Breaking text into words
- Stopword Removal: Remove common words (the, is, at)
- Stemming: Reduce words to root form (running → run)
- TF-IDF: Term Frequency-Inverse Document Frequency

### Machine Learning Concepts
- **Bias**: How well model fits training data
- **Variance**: How well model generalizes to new data
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of Precision and Recall

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more ML models (XGBoost, LGBM, Neural Networks)
- Implement cross-validation
- Add hyperparameter tuning
- Support multiple languages
- Add data augmentation
- Implement active learning

## 📧 Support

If you encounter issues or have questions, please check the troubleshooting section or create an issue.

---

**Happy Analyzing! 🎉**
