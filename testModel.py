#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example script for using the saved email classification model
"""

import pandas as pd
import pickle
import os
from trainModel import *
def load_model(model_path='email_classifier_model.pkl', pipeline_path='email_classifier_pipeline.pkl'):
    """Load the trained model and preprocessing pipeline."""
    print(f"Loading model from {model_path}...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading pipeline from {pipeline_path}...")
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    return model, pipeline


def predict_single_email(email_data, model, pipeline):
    """
    Make a prediction for a single email.
    
    Args:
        email_data (dict): Dictionary containing email fields
        model: Trained XGBoost model
        pipeline: Preprocessing pipeline
    
    Returns:
        Prediction label
    """
    # Convert to DataFrame if it's a dictionary
    if isinstance(email_data, dict):
        email_df = pd.DataFrame([email_data])
    else:
        email_df = email_data
    
    # Process with the pipeline
    processed_data = pipeline.transform(email_df)
    
    # Make prediction
    prediction = model.predict(processed_data)
    
    return prediction[0]


def batch_predict(emails_df, model, pipeline):
    """
    Make predictions for multiple emails.
    
    Args:
        emails_df (DataFrame): DataFrame containing multiple emails
        model: Trained XGBoost model
        pipeline: Preprocessing pipeline
    
    Returns:
        Array of predictions
    """
    # Process with the pipeline
    processed_data = pipeline.transform(emails_df)
    
    # Make predictions
    predictions = model.predict(processed_data)
    
    return predictions


def main():
    """Main function to demonstrate model usage."""
    # Load the model and pipeline
    model, pipeline = load_model()
    
    # Example 1: Single email prediction
    print("\n--- Example 1: Single Email Prediction ---")
    
    test_email = {
    'sender': 'it-support@example.com',
    'date': 'Mon, 14 Apr 2025 10:30:45 +0000',
    'subject': 'URGENT: Password Expiration Notice',
    'body': (
        'Dear User,\n\n'
        'Our records indicate that your email password is set to expire today. '
        'To maintain uninterrupted access to your account, please verify your credentials immediately by clicking the link below:\n\n'
        'https://example.com/verify-account\n\n'
        'Failure to do so may result in temporary suspension of your email services.\n\n'
        'Thank you for your prompt attention to this matter.\n\n'
        'Best regards,\n'
        'IT Support Team'
    ),
    'urls': None
}

    
    prediction = predict_single_email(test_email, model, pipeline)
    print(f"Prediction for test email: {prediction}")
    
    # Example 2: Batch prediction (if you have a CSV)
    print("\n--- Example 2: Batch Prediction ---")
    try:
        # Load a test CSV file if available
        test_csv_path = 'test_emails.csv'
        if os.path.exists(test_csv_path):
            test_df = pd.read_csv(test_csv_path)
            batch_predictions = batch_predict(test_df, model, pipeline)
            print(f"Batch predictions: {batch_predictions[:5]}...")  # Show first 5 predictions
            
            # Save predictions
            test_df['predicted_label'] = batch_predictions
            test_df.to_csv('test_emails_with_predictions.csv', index=False)
            print(f"Predictions saved to 'test_emails_with_predictions.csv'")
        else:
            print(f"Test file {test_csv_path} not found. Skipping batch prediction.")
    except Exception as e:
        print(f"Error in batch prediction: {e}")
    
    print("\nEmail Classification Model Usage Complete!")


if __name__ == "__main__":
    main()