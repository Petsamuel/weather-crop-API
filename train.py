import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define crop dictionary at module level
CROP_DICT = {
    'rice': 1, 'maize': 2, 'chickpea': 3, 'kidneybeans': 4,
    'pigeonpeas': 5, 'mothbeans': 6, 'mungbean': 7, 'blackgram': 8,
    'lentil': 9, 'pomegranate': 10, 'banana': 11, 'mango': 12,
    'grapes': 13, 'watermelon': 14, 'muskmelon': 15, 'apple': 16,
    'orange': 17, 'papaya': 18, 'coconut': 19, 'cotton': 20,
    'jute': 21, 'coffee': 22
}

def load_and_preprocess_data():
    try:
        data = pd.read_csv('crop_recommendation.csv')
        data['label'] = data['label'].map(CROP_DICT)
        X = data.drop('label', axis=1)
        y = data['label']
        return X, y
    except Exception as e:
        logging.error(f"Error loading or preprocessing data: {e}")
        raise

def evaluate_classifiers(X_train, X_test, y_train, y_test):
    # Define classifiers
    classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    # Evaluate each classifier
    results = {}
    for name, clf in classifiers.items():
        # Train and predict
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Store results
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_scores': cross_val_score(clf, X_train, y_train, cv=5),
            'report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'model': clf
        }
        
    return results

def plot_classifier_comparison(results):
    accuracies = [v['accuracy'] for v in results.values()]
    names = list(results.keys())
    
    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies)
    plt.title('Classifier Comparison')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('classifier_comparison.png')

def plot_confusion_matrices(results, y_test):
    for name, metrics in results.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'confusion_matrix_{name}.png')

def plot_cv_scores(results):
    plt.figure(figsize=(10, 6))
    for name, metrics in results.items():
        plt.plot(metrics['cv_scores'], label=name)
    plt.title('Cross-Validation Scores')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cv_scores.png')

def plot_feature_importance(results, X):
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])[1]['model']
    if hasattr(best_clf, 'feature_importances_'):
        importances = best_clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.title('Feature Importances')
        plt.bar(range(X.shape[1]), importances[indices], align='center')
        plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
        plt.tight_layout()
        plt.savefig('feature_importances.png')

def plot_pairplot(X, y):
    df = X.copy()
    df['label'] = y
    sns.pairplot(df, hue='label', diag_kind='kde')
    plt.tight_layout()
    plt.savefig('pairplot.png')

def plot_correlation_heatmap(X):
    plt.figure(figsize=(12, 8))
    sns.heatmap(X.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')

def train_and_evaluate(test_size=0.2, random_state=42):
    # Load and scale data
    X, y = load_and_preprocess_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Evaluate classifiers
    results = evaluate_classifiers(X_train, X_test, y_train, y_test)
    
    # Plot results
    plot_classifier_comparison(results)
    plot_confusion_matrices(results, y_test)
    plot_cv_scores(results)
    plot_feature_importance(results, X)
    plot_pairplot(pd.DataFrame(X_scaled, columns=X.columns), y)
    plot_correlation_heatmap(pd.DataFrame(X_scaled, columns=X.columns))
    
    # Get best model
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    # Save best model
    joblib.dump(best_clf[1]['model'], 'best_crop_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return results

def test_crop_prediction(temperature, humidity, rainfall, N, P, K, ph):
    # Load model and scaler
    model = joblib.load('best_crop_model.pkl')
    scaler = joblib.load('scaler.pkl')
    
    # Prepare input data
    input_data = pd.DataFrame([{
        'N': N,
        'P': P,
        'K': K,
        'temperature': temperature,
        'humidity': humidity,
        'ph': ph,
        'rainfall': rainfall
    }])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Get predictions
    predictions = model.predict(input_scaled)
    
    # Map predictions to crop names
    crop_dict = {
        1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans',
        5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean', 8: 'blackgram',
        9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
        13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple',
        17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
        21: 'jute', 22: 'coffee'
    }
    
    recommended_crops = [crop_dict.get(pred, "Unknown Crop") for pred in predictions]
    return recommended_crops

if __name__ == "__main__":
    results = train_and_evaluate()
    temperature = 25.0
    humidity = 71.0
    rainfall = 103.0
    N = 90
    P = 42
    K = 43
    ph = 6.5
    crop = test_crop_prediction(temperature, humidity, rainfall, N, P, K, ph)
    print(f"Recommended crops: {crop}")
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"CV Scores: {metrics['cv_scores'].mean():.3f} (+/- {metrics['cv_scores'].std() * 2:.3f})")
        print(f"Classification Report:\n{metrics['report']}")