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
from sklearn.metrics import accuracy_score, classification_report
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

def train_and_evaluate(test_size=0.2, random_state=42):
    # Load and scale data
    X, y = load_and_preprocess_data()
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CROP_DICT.keys(), yticklabels=CROP_DICT.keys())
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    
    # Save model artifacts
    joblib.dump(model, 'crop_recommendation_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return model, cv_scores, classification_report(y_test, y_pred), model.feature_importances_

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
    
    # Get best model
    best_clf = max(results.items(), key=lambda x: x[1]['accuracy'])
    
    # Save best model
    joblib.dump(best_clf[1]['model'], 'best_crop_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return results

if __name__ == "__main__":
    results = train_and_evaluate()
    for name, metrics in results.items():
        print(f"\n{name} Results:")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"CV Scores: {metrics['cv_scores'].mean():.3f} (+/- {metrics['cv_scores'].std() * 2:.3f})")
        print(f"Classification Report:\n{metrics['report']}")