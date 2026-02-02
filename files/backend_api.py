"""
Restaurant Review Sentiment Analysis - Flask Backend
Provides REST API endpoints for the NLP dashboard
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

# Download required NLTK data (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Global variables for model and vectorizer
vectorizer = None
model = None
X_train = None
y_train = None
X_test = None
y_test = None
dataset = None

class SentimentAnalyzer:
    """
    Main class for sentiment analysis operations
    """
    
    def __init__(self, data_path='Restaurant_Reviews.tsv'):
        self.data_path = data_path
        self.vectorizer = TfidfVectorizer(max_features=1500)
        self.ps = PorterStemmer()
        self.corpus = []
        self.dataset = None
        self.models = {}
        
    def load_data(self):
        """Load the restaurant reviews dataset"""
        try:
            self.dataset = pd.read_csv(self.data_path, delimiter='\t', quoting=3)
            print(f"✓ Dataset loaded: {len(self.dataset)} reviews")
            return True
        except FileNotFoundError:
            print(f"✗ Error: File '{self.data_path}' not found")
            return False
        
    def preprocess_text(self, text):
        """
        Clean and preprocess a single text
        Steps: Remove special chars → Lowercase → Tokenize → Remove stopwords → Stem
        """
        # Remove special characters
        review = re.sub('[^a-zA-Z]', ' ', text)
        
        # Convert to lowercase
        review = review.lower()
        
        # Tokenize
        review = review.split()
        
        # Remove stopwords and stem
        review = [self.ps.stem(word) for word in review 
                 if not word in set(stopwords.words('english'))]
        
        # Join back
        review = ' '.join(review)
        
        return review
    
    def create_corpus(self):
        """Create corpus from all reviews"""
        print("Processing reviews...")
        self.corpus = []
        
        for i in range(len(self.dataset)):
            review_text = self.dataset['Review'][i]
            processed_review = self.preprocess_text(review_text)
            self.corpus.append(processed_review)
            
        print(f"✓ Corpus created: {len(self.corpus)} processed reviews")
        
    def vectorize(self):
        """Convert corpus to TF-IDF features"""
        X = self.vectorizer.fit_transform(self.corpus).toarray()
        y = self.dataset.iloc[:, 1].values
        
        print(f"✓ Features extracted: {X.shape[1]} features")
        
        return X, y
    
    def train_models(self, X, y, test_size=0.20):
        """Train multiple ML models and compare performance"""
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.svm import SVC
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=0
        )
        
        print(f"\n{'='*60}")
        print(f"Training Models (Train: {len(X_train)}, Test: {len(X_test)})")
        print(f"{'='*60}")
        
        # Define models
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=0),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=0),
            'Logistic Regression': LogisticRegression(random_state=0, max_iter=1000),
            'Naive Bayes': MultinomialNB(),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'SVM': SVC(kernel='linear', random_state=0)
        }
        
        results = []
        
        for name, clf in models.items():
            # Train
            clf.fit(X_train, y_train)
            
            # Predict
            y_pred = clf.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Bias and Variance
            train_score = clf.score(X_train, y_train)
            test_score = clf.score(X_test, y_test)
            
            results.append({
                'name': name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'bias': train_score,
                'variance': test_score
            })
            
            # Store model
            self.models[name] = clf
            
            print(f"\n{name}:")
            print(f"  Accuracy:  {accuracy:.3f}")
            print(f"  Precision: {precision:.3f}")
            print(f"  Recall:    {recall:.3f}")
            print(f"  F1 Score:  {f1:.3f}")
            print(f"  Bias:      {train_score:.3f}")
            print(f"  Variance:  {test_score:.3f}")
        
        print(f"\n{'='*60}\n")
        
        return results, X_train, X_test, y_train, y_test
    
    def predict_sentiment(self, text, model_name='Decision Tree'):
        """Predict sentiment for a new review"""
        # Preprocess
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        features = self.vectorizer.transform([processed_text]).toarray()
        
        # Predict
        model = self.models.get(model_name, self.models['Decision Tree'])
        prediction = model.predict(features)[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features)[0]
            confidence = max(probabilities)
        else:
            confidence = 0.85  # Default confidence for models without probability
        
        return {
            'sentiment': 'Positive' if prediction == 1 else 'Negative',
            'prediction': int(prediction),
            'confidence': float(confidence),
            'model_used': model_name
        }
    
    def save_model(self, model_name='model.pkl', vectorizer_name='vectorizer.pkl'):
        """Save trained model and vectorizer"""
        with open(model_name, 'wb') as f:
            pickle.dump(self.models['Decision Tree'], f)
        
        with open(vectorizer_name, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"✓ Model saved: {model_name}")
        print(f"✓ Vectorizer saved: {vectorizer_name}")
    
    def load_model(self, model_name='model.pkl', vectorizer_name='vectorizer.pkl'):
        """Load pre-trained model and vectorizer"""
        try:
            with open(model_name, 'rb') as f:
                self.models['Decision Tree'] = pickle.load(f)
            
            with open(vectorizer_name, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print(f"✓ Model loaded: {model_name}")
            print(f"✓ Vectorizer loaded: {vectorizer_name}")
            return True
        except FileNotFoundError:
            print("✗ Model files not found. Train model first.")
            return False


# Initialize analyzer
analyzer = SentimentAnalyzer()

# Store model results globally
model_results = []
confusion_mat = None


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'NLP Sentiment Analysis API is running'
    })


@app.route('/api/train', methods=['POST'])
def train_model():
    """
    Train the sentiment analysis model
    Expects: { "data_path": "path/to/data.tsv" } (optional)
    """
    global model_results, confusion_mat, X_test, y_test
    
    try:
        data = request.json or {}
        data_path = data.get('data_path', 'Restaurant_Reviews.tsv')
        
        # Update data path if provided
        analyzer.data_path = data_path
        
        # Load data
        if not analyzer.load_data():
            return jsonify({
                'error': 'Failed to load dataset',
                'message': f'File not found: {data_path}'
            }), 404
        
        # Create corpus
        analyzer.create_corpus()
        
        # Vectorize
        X, y = analyzer.vectorize()
        
        # Train models
        results, X_train, X_test_local, y_train, y_test_local = analyzer.train_models(X, y)
        model_results = results
        X_test = X_test_local
        y_test = y_test_local
        
        # Get confusion matrix for Decision Tree
        dt_model = analyzer.models['Decision Tree']
        y_pred = dt_model.predict(X_test)
        confusion_mat = confusion_matrix(y_test, y_pred)
        
        # Save model
        analyzer.save_model()
        
        return jsonify({
            'message': 'Models trained successfully',
            'models': results,
            'confusion_matrix': confusion_mat.tolist(),
            'dataset_size': len(analyzer.dataset),
            'feature_count': X.shape[1]
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Training failed',
            'message': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict sentiment for a review
    Expects: { "text": "review text", "model": "model_name" }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        model_name = data.get('model', 'Decision Tree')
        
        # Check if model is trained
        if not analyzer.models:
            # Try to load saved model
            if not analyzer.load_model():
                return jsonify({
                    'error': 'Model not trained',
                    'message': 'Please train the model first using /api/train'
                }), 400
        
        # Predict
        result = analyzer.predict_sentiment(text, model_name)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e)
        }), 500


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get performance comparison of all models"""
    global model_results
    
    if not model_results:
        return jsonify({
            'error': 'No model results available',
            'message': 'Train models first using /api/train'
        }), 400
    
    return jsonify({
        'models': model_results
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get detailed metrics including confusion matrix"""
    global confusion_mat, model_results
    
    if confusion_mat is None:
        return jsonify({
            'error': 'No metrics available',
            'message': 'Train models first using /api/train'
        }), 400
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = confusion_mat.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return jsonify({
        'confusion_matrix': {
            'true_positive': int(tp),
            'false_positive': int(fp),
            'true_negative': int(tn),
            'false_negative': int(fn)
        },
        'metrics': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'specificity': float(specificity)
        }
    })


@app.route('/api/dataset-info', methods=['GET'])
def get_dataset_info():
    """Get information about the dataset"""
    if analyzer.dataset is None:
        return jsonify({
            'error': 'Dataset not loaded',
            'message': 'Train model first using /api/train'
        }), 400
    
    positive_reviews = analyzer.dataset[analyzer.dataset.iloc[:, 1] == 1].shape[0]
    negative_reviews = analyzer.dataset[analyzer.dataset.iloc[:, 1] == 0].shape[0]
    
    return jsonify({
        'total_reviews': len(analyzer.dataset),
        'positive_reviews': int(positive_reviews),
        'negative_reviews': int(negative_reviews),
        'balance_ratio': float(positive_reviews / len(analyzer.dataset))
    })


@app.route('/api/preprocess', methods=['POST'])
def preprocess_text():
    """
    Preprocess text without prediction
    Expects: { "text": "review text" }
    """
    try:
        data = request.json
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text'
            }), 400
        
        text = data['text']
        processed = analyzer.preprocess_text(text)
        
        return jsonify({
            'original': text,
            'processed': processed,
            'word_count': len(processed.split())
        })
        
    except Exception as e:
        return jsonify({
            'error': 'Preprocessing failed',
            'message': str(e)
        }), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Restaurant Review Sentiment Analysis API")
    print("="*60)
    print("\nAvailable Endpoints:")
    print("  GET  /api/health          - Health check")
    print("  POST /api/train           - Train models")
    print("  POST /api/predict         - Predict sentiment")
    print("  GET  /api/models          - Get model comparison")
    print("  GET  /api/metrics         - Get performance metrics")
    print("  GET  /api/dataset-info    - Get dataset information")
    print("  POST /api/preprocess      - Preprocess text")
    print("\n" + "="*60 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)