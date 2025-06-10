#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Email Classification Model with Multiple Models and Model Persistence
"""

# Standard libraries
import os
import re
import pickle
from datetime import datetime

# Data processing libraries
import numpy as np
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns
# from wordcloud import WordCloud

# Machine learning libraries
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


# Define custom transformers
class DropNaBodyTransformer(BaseEstimator, TransformerMixin):
    """Transformer to drop rows with missing body content."""
    
    def __init__(self, column="body"):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        return X.dropna(subset=[self.column])


class MissingValueIndicatorFiller(BaseEstimator, TransformerMixin):
    """Transformer to fill missing values and add indicator columns."""
    
    def __init__(self, fill_value="unknown"):
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        
        for feature in X.columns:
            # Create the indicator column
            indicator_col = f"{feature}_is_known"
            X[indicator_col] = X[feature].notna().astype(int)
            
            # Fill missing values
            X[feature] = X[feature].fillna(self.fill_value)
        
        return X


class URLImputer(BaseEstimator, TransformerMixin):
    """Transformer to extract URL presence from body text."""
    
    def __init__(self, url_regex=None):
        self.url_regex = url_regex or r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)'
        self.url_finder = re.compile(self.url_regex)
        
    def fit(self, X, y=None):
        return self

    def _identify_url(self, row):
        if pd.isna(row['urls']):
            body = str(row['body'])
            is_url = bool(self.url_finder.search(body))
            return int(is_url)
        else:
            return row['urls']

    def transform(self, X, y=None):
        impute_X = X.copy()
        impute_X['urls'] = X.apply(self._identify_url, axis=1)
        return impute_X


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer to extract meaningful features from date strings."""
    
    def __init__(self, placeholder="unknown", date_format="%a, %d %b %Y %H:%M:%S %z"):
        self.placeholder = placeholder
        self.date_format = date_format

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Handle DataFrame or numpy array input
        if isinstance(X, pd.DataFrame):
            date_column = X.iloc[:, 0]  # Assuming date is the first column
        else:
            date_column = X[:, 0]
        
        # Parse dates and extract features
        features = []
        for date_str in date_column:
            date_obj = self._safe_parse_date(date_str)
            
            if date_obj:
                day_of_week = date_obj.weekday()
                hour_of_day = date_obj.hour
                is_weekend = int(day_of_week >= 5)
                year = date_obj.year
                month = date_obj.month
                day = date_obj.day
            else:
                day_of_week = 0
                hour_of_day = 0
                is_weekend = 0
                year = 1900
                month = 1
                day = 1
                
            features.append([day_of_week, hour_of_day, is_weekend, year, month, day])
        
        return np.array(features)

    def _safe_parse_date(self, date_str):
        try:
            return datetime.strptime(str(date_str), self.date_format)
        except (ValueError, TypeError):
            return None


def load_data(file_paths=None):
    """Load and combine data from multiple CSV files."""
    if not file_paths:
        # If no paths provided, search for CSV files
        file_paths = []
        for dirname, _, filenames in os.walk('./data'):
            for filename in filenames:
                if filename.endswith('.csv'):
                    file_paths.append(os.path.join(dirname, filename))
    
    print(f"Loading data from {len(file_paths)} files...")
    
    dataframes = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            dataframes.append(df)
            print(f"Loaded: {path} - {df.shape[0]} rows")
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    if not dataframes:
        raise ValueError("No data loaded. Check file paths.")
    
    # Combine dataframes and clean up
    combined_df = pd.concat(dataframes, axis=0, ignore_index=True)
    combined_df = combined_df.dropna(subset=['body'])
    combined_df.reset_index(drop=True, inplace=True)
    
    print(f"Combined data shape: {combined_df.shape}")
    return combined_df


def create_pipeline():
    """Create the full preprocessing and feature extraction pipeline."""
    # Data preprocessing pipeline
    data_preprocessing = ColumnTransformer([
        ("fill_na", MissingValueIndicatorFiller(), ["subject", "sender", "date"]),
        ("url", URLImputer(), ["urls", "body"])
    ])
    
    datapreprocessing_pipeline = Pipeline([
        ("handleNA", data_preprocessing)
    ])
    
    # Feature extraction pipeline
    text_feature_extraction = ColumnTransformer([
        ("subject_tfidf", TfidfVectorizer(max_features=5000), 0),  # 'subject'
        ("sender_tfidf", TfidfVectorizer(max_features=1000), 1),   # 'sender'
        ("body_tfidf", TfidfVectorizer(max_features=10000), 2)     # 'body'
    ], remainder="passthrough")
    
    feature_extraction_pipeline = ColumnTransformer([
        ("date_extraction", DateFeatureExtractor(), [2]),  # Extract from 'date'
        ("text_extraction", text_feature_extraction, [0, 1, 7])  # Process text columns
    ], remainder='passthrough')
    
    # Combine into a full pipeline
    full_pipeline = Pipeline([
        ("preprocessing", datapreprocessing_pipeline),
        ("feature_extraction", feature_extraction_pipeline)
    ])
    
    return full_pipeline


def create_ann_model(input_dim):
    """
    Create an Artificial Neural Network for classification
    """
    def build_model():
        model = Sequential()
        model.add(Dense(256, input_dim=input_dim, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(
            loss='binary_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy']
        )
        return model
    
    return build_model


def train_and_evaluate_models(X_train, X_test, y_train, y_test, pipeline):
    """Train and evaluate multiple models."""
    print("Preprocessing data...")
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)
    
    print(f"Processed data shape: {X_train_processed.shape}")
    
    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "XGBoost": XGBClassifier(
            random_state=42,
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
    }
    
    # Train and evaluate traditional models
    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Cross-validation for F1
        cv_f1_scores = cross_val_score(model, X_train_processed, y_train, cv=3, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_f1_scores}")
        print(f"Mean CV F1 score: {cv_f1_scores.mean():.4f}")
        
        # Cross-validation for Accuracy
        cv_acc_scores = cross_val_score(model, X_train_processed, y_train, cv=3, scoring='accuracy')
        print(f"Cross-validation Accuracy scores: {cv_acc_scores}")
        print(f"Mean CV Accuracy score: {cv_acc_scores.mean():.4f}")
        
        # Fit on the full training set
        model.fit(X_train_processed, y_train)
        
        # Evaluate on test set
        y_pred = model.predict(X_test_processed)
        test_f1 = f1_score(y_test, y_pred)
        test_acc = accuracy_score(y_test, y_pred)
        
        print(f"Test F1 score: {test_f1:.4f}")
        print(f"Test Accuracy score: {test_acc:.4f}")
        
        # Store results
        results[name] = {
            'model': model,
            'cv_f1_mean': cv_f1_scores.mean(),
            'cv_accuracy_mean': cv_acc_scores.mean(),
            'test_f1': test_f1,
            'test_accuracy': test_acc
        }
    
    # Train and evaluate ANN model
    print("\nTraining Neural Network...")
    input_dim = X_train_processed.shape[1]
    
    # Create the ANN model directly without scikit-learn wrapper
    build_fn = create_ann_model(input_dim)
    
    # Define early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Create a TensorFlow model directly for training
    tf_model = build_fn()
    
    # Convert data and labels to numpy arrays if they aren't already
    X_train_arr = X_train_processed.toarray() if hasattr(X_train_processed, 'toarray') else X_train_processed
    X_test_arr = X_test_processed.toarray() if hasattr(X_test_processed, 'toarray') else X_test_processed
    
    # Train the model
    history = tf_model.fit(
        X_train_arr,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate on test set
    ann_scores = tf_model.evaluate(X_test_arr, y_test, verbose=0)
    test_loss, test_accuracy = ann_scores
    
    # Predictions for F1 score
    y_pred_proba = tf_model.predict(X_test_arr)
    y_pred = (y_pred_proba > 0.5).astype(int)
    test_f1 = f1_score(y_test, y_pred)
    
    print(f"Neural Network Test Loss: {test_loss:.4f}")
    print(f"Neural Network Test Accuracy: {test_accuracy:.4f}")
    print(f"Neural Network Test F1 score: {test_f1:.4f}")
    
    # Store ANN results
    results["Neural Network"] = {
        'model': tf_model,
        'cv_f1_mean': None,  # ANN doesn't use sklearn's CV
        'cv_accuracy_mean': None,
        'test_f1': test_f1,
        'test_accuracy': test_accuracy,
        'history': history.history
    }
    
    # Print classification report for the best model (highest test F1)
    best_model_name = max(results, key=lambda x: results[x]['test_f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest model based on F1 Score: {best_model_name}")
    
    if best_model_name == "Neural Network":
        y_pred = (best_model.predict(X_test_arr) > 0.5).astype(int)
    else:
        y_pred = best_model.predict(X_test_processed)
    
    print("\nClassification Report for Best Model:")
    print(classification_report(y_test, y_pred))
    
    # Create confusion matrix for the best model
    create_confusion_matrix(y_test, y_pred, best_model_name)
    
    # Compare model performances
    compare_models(results)
    
    return results, pipeline


def create_confusion_matrix(y_true, y_pred, model_name):
    """Create and save confusion matrix visualization."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
    plt.close()


def compare_models(results):
    """Create comparison visualizations of model performances."""
    # Extract metrics
    model_names = list(results.keys())
    f1_scores = [results[model]['test_f1'] for model in model_names]
    acc_scores = [results[model]['test_accuracy'] for model in model_names]
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(model_names))
    width = 0.35
    
    ax.bar(x - width/2, f1_scores, width, label='F1 Score')
    ax.bar(x + width/2, acc_scores, width, label='Accuracy')
    
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    # If Neural Network was trained, plot its training history
    if "Neural Network" in results and results["Neural Network"]["history"] is not None:
        history = results["Neural Network"]["history"]
        
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Neural Network Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['accuracy'], label='Training Accuracy')
        plt.plot(history['val_accuracy'], label='Validation Accuracy')
        plt.title('Neural Network Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('neural_network_training.png')
        plt.close()


def save_model(model, pipeline, model_path='best_email_classifier_model.pkl', pipeline_path='email_classifier_pipeline.pkl'):
    """Save the trained model and preprocessing pipeline."""
    print(f"Saving model to {model_path}...")
    
    # Handle TensorFlow models differently
    if isinstance(model, tf.keras.Model):
        model.save(model_path.replace('.pkl', '.h5'))
    else:
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
    
    print(f"Saving pipeline to {pipeline_path}...")
    with open(pipeline_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print("Model and pipeline saved successfully!")


def load_model(model_path='best_email_classifier_model.pkl', pipeline_path='email_classifier_pipeline.pkl'):
    """Load the trained model and preprocessing pipeline."""
    print(f"Loading model from {model_path}...")
    
    # Check if it's a TensorFlow model
    if model_path.endswith('.h5'):
        from tensorflow.keras.models import load_model as load_tf_model
        model = load_tf_model(model_path)
    else:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    
    print(f"Loading pipeline from {pipeline_path}...")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    return model, pipeline


def predict_email(email_data, model, pipeline, is_neural_network=False):
    """Make predictions on new email data."""
    if isinstance(email_data, dict):
        # Convert dictionary to DataFrame
        email_data = pd.DataFrame([email_data])
    
    # Preprocess the data
    X_processed = pipeline.transform(email_data)
    
    # Make prediction based on model type
    if is_neural_network:
        X_processed_arr = X_processed.toarray() if hasattr(X_processed, 'toarray') else X_processed
        prediction_proba = model.predict(X_processed_arr)
        prediction = (prediction_proba > 0.5).astype(int)
    else:
        prediction = model.predict(X_processed)
    
    return prediction


def main():
    """Main function to run the email classification pipeline."""
    # Load data
    df = load_data()
    print(f"Data loaded: {df.shape}")
    
    # Display data information
    print("\nData Overview:")
    print(df.info())
    print("\nTarget distribution:")
    print(df["label"].value_counts())
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop("label", axis=1), 
        df["label"], 
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    
    # Create pipeline
    pipeline = create_pipeline()
    
    # Train and evaluate models
    results, pipeline = train_and_evaluate_models(X_train, X_test, y_train, y_test, pipeline)
    
    # Find the best model based on F1 score
    best_model_name = max(results, key=lambda x: results[x]['test_f1'])
    best_model = results[best_model_name]['model']
    
    print(f"\nBest performing model: {best_model_name}")
    print(f"F1 Score: {results[best_model_name]['test_f1']:.4f}")
    print(f"Accuracy: {results[best_model_name]['test_accuracy']:.4f}")
    
    # Save best model and pipeline
    is_neural_network = best_model_name == "Neural Network"
    model_path = 'best_email_classifier_model.h5' if is_neural_network else 'best_email_classifier_model.pkl'
    save_model(best_model, pipeline, model_path=model_path)
    
    print("\nEmail Classification Model Training Complete!")
    
    # Example of how to use the model for prediction
    print("\nExample prediction:")
    sample_email = {
        'subject': 'Meeting tomorrow',
        'sender': 'colleague@company.com',
        'date': 'Mon, 20 Apr 2025 10:30:00 +0000',
        'body': 'Hi team, just a reminder about our meeting tomorrow at 2pm.',
        'urls': None
    }
    
    prediction = predict_email(sample_email, best_model, pipeline, is_neural_network)
    print(f"Prediction: {prediction[0]}")


if __name__ == "__main__":
    main()