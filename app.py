
"""
Enhanced Phishing Email Detector - Streamlit App with VirusTotal Integration
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import requests
import time
import io
import tempfile
import os
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin

# --- Custom Transformer Classes ---
# These need to be defined here so pickle can deserialize the model correctly
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


# --- VirusTotal API Functions ---
def upload_file_to_virustotal(file_bytes, api_key):
    """Upload a file to VirusTotal for scanning."""
    url = 'https://www.virustotal.com/api/v3/files'
    headers = {
        'x-apikey': api_key
    }
    
    files = {'file': ('file', file_bytes)}
    
    with st.spinner('Uploading file to VirusTotal...'):
        response = requests.post(url, headers=headers, files=files)
    
    if response.status_code == 200:
        return response.json()['data']['id']
    else:
        st.error(f"Error uploading file to VirusTotal: {response.status_code}")
        st.error(response.text)
        return None


def get_virustotal_analysis(analysis_id, api_key, max_wait_time=60):
    """Get analysis report from VirusTotal with timeout."""
    url = f'https://www.virustotal.com/api/v3/analyses/{analysis_id}'
    headers = {
        'x-apikey': api_key
    }
    
    start_time = time.time()
    with st.spinner('Waiting for VirusTotal analysis results...'):
        while (time.time() - start_time) < max_wait_time:
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                analysis = response.json()
                status = analysis['data']['attributes']['status']
                if status == 'completed':
                    return analysis
                
                # Check if the analysis is taking too long
                elapsed = time.time() - start_time
                if elapsed > max_wait_time / 2:
                    st.info(f"Analysis in progress ({elapsed:.0f}s elapsed)... This might take a while.")
            
            time.sleep(5)
    
    st.warning("VirusTotal analysis is taking longer than expected. Results will be partial.")
    return response.json() if response.status_code == 200 else None


def display_virustotal_results(report):
    """Display VirusTotal analysis results in a user-friendly format."""
    if not report:
        st.error("Unable to retrieve VirusTotal analysis results.")
        return
    
    # Extract statistics
    stats = report['data']['attributes']['stats']
    total_engines = sum(stats.values())
    
    # Create columns for stats
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Malicious", stats['malicious'])
    with col2:
        st.metric("Suspicious", stats['suspicious'])
    with col3:
        st.metric("Harmless", stats['harmless'])
    with col4:
        st.metric("Undetected", stats['undetected'])
    with col5:
        st.metric("Timeout", stats.get('timeout', 0))
    
    # Calculate threat score
    threat_score = (stats['malicious'] * 100 + stats['suspicious'] * 50) / total_engines
    threat_level = "High" if threat_score > 30 else "Medium" if threat_score > 10 else "Low"
    
    # Display threat assessment
    threat_color = "red" if threat_level == "High" else "orange" if threat_level == "Medium" else "green"
    st.markdown(f"<h3 style='color: {threat_color}'>Threat Level: {threat_level} ({threat_score:.1f}%)</h3>", unsafe_allow_html=True)
    
    # Show detailed results in an expandable section
    with st.expander("Detailed Scan Results"):
        # Create a table of AV results
        results = report['data']['attributes']['results']
        
        # Convert to DataFrame for better display
        results_data = []
        for engine, result in results.items():
            category = result.get('category', 'unknown')
            result_name = result.get('result', 'N/A')
            
            # Determine status color
            if category == 'malicious':
                status = "⚠️ " + result_name
            elif category == 'suspicious':
                status = "⚠️ " + result_name
            elif category == 'harmless':
                status = "✅ Clean"
            else:
                status = "ℹ️ " + category.capitalize()
                
            results_data.append({
                "Engine": engine,
                "Category": category.capitalize(),
                "Result": status
            })
        
        # Display as a DataFrame
        if results_data:
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True)


# --- App Functions ---
@st.cache_resource
def load_model_and_pipeline():
    """Load the model and preprocessing pipeline with caching."""
    try:
        with open('email_classifier_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('email_classifier_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
        return model, pipeline, True
    except FileNotFoundError as e:
        st.error(f"Error loading files: {e}")
        return None, None, False


def format_prediction(prediction, confidence):
    """Format the prediction result with confidence score."""
    if prediction == 1:
        return f"⚠️ PHISHING EMAIL DETECTED ⚠️ (Confidence: {confidence:.2f}%)"
    else:
        return f"✅ Email appears legitimate (Confidence: {confidence:.2f}%)"


def get_prediction_label(prediction):
    """Return a human-readable prediction label."""
    return "PHISHING" if prediction == 1 else "LEGITIMATE"


def process_input(sender, date, subject, body, urls_present):
    """Process user inputs into the format expected by the model."""
    # Create a DataFrame with the input
    input_data = pd.DataFrame({
        'sender': [sender],
        'date': [date],
        'subject': [subject],
        'body': [body],
        'urls': [urls_present]
    })
    return input_data


def analyze_email_text(body):
    """Analyze email body for potential phishing indicators."""
    indicators = []
    
    # Common phishing phrases
    urgent_phrases = ['urgent', 'immediate action', 'account suspended', 'verify your', 
                      'security alert', 'unauthorized access', 'limited time']
    
    # Check for urgency language
    if any(phrase in body.lower() for phrase in urgent_phrases):
        indicators.append("Contains urgent language")
    
    # Check for URLs that don't match their anchor text
    url_pattern = re.compile(r'<a\s+(?:[^>]*?\s+)?href=(["\'])(.*?)\1.*?>(.*?)<\/a>')
    matches = url_pattern.findall(body)
    for match in matches:
        href, text = match[1], match[2] if len(match) > 2 else ""
        if href not in text and "http" in href:
            indicators.append("Contains misleading links")
    
    # Check for misspelled domain names (simplistic approach)
    domains = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook']
    for domain in domains:
        misspelled = [f"{domain[:-1]}p{domain[-1:]}", f"{domain[0]}{domain[1:].replace('o', '0')}"]
        if any(misspell in body.lower() for misspell in misspelled):
            indicators.append("Contains misspelled domain names")
            break
    
    # Check for suspicious attachments
    if any(ext in body.lower() for ext in ['.exe', '.zip', '.jar', '.scr']):
        indicators.append("References suspicious file attachments")
    
    # Check for personal information requests
    if any(info in body.lower() for info in ['ssn', 'social security', 'password', 'credit card',
                                           'bank account', 'login']):
        indicators.append("Requests personal information")
    
    return indicators


def display_email_details(sender, date, subject, body, indicators):
    """Display email details with highlighted indicators."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Email Metadata")
        st.write(f"**From:** {sender}")
        st.write(f"**Date:** {date}")
        st.write(f"**Subject:** {subject}")
        
        if indicators:
            st.subheader("Suspicious Indicators")
            for indicator in indicators:
                st.warning(indicator)
    
    with col2:
        st.subheader("Email Body")
        st.text_area("", body, height=200, disabled=True)


def main():
    """Main function for the Streamlit app."""
    # Page configuration
    st.set_page_config(
        page_title="Advanced Phishing Email Detector",
        page_icon="🔍",
        layout="wide"
    )
    
    # Initialize session state for VirusTotal API key
    if 'virustotal_api_key' not in st.session_state:
        st.session_state.virustotal_api_key = "YOUR VIRUS TOTAL API KEY HERE"
    
    # Load model and pipeline
    model, pipeline, model_loaded = load_model_and_pipeline()
    
    # App title and description
    st.title("🛡️ Advanced Phishing Email Detector")
    st.write("Enter email details and check attachments for potential security threats.")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    # VirusTotal API Key input in sidebar
    virustotal_api_key = st.sidebar.text_input(
        "VirusTotal API Key", 
        type="password",
        help="Enter your VirusTotal API key to enable attachment scanning",
        value=st.session_state.virustotal_api_key
    )
    
    # Save API key to session state
    if virustotal_api_key != st.session_state.virustotal_api_key:
        st.session_state.virustotal_api_key = virustotal_api_key
    
    # Input form
    with st.form("email_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            sender = st.text_input("Sender Email Address", 
                                  placeholder="example@domain.com")
            date_input = st.date_input("Date Received")
            time_input = st.time_input("Time Received")
            
        with col2:
            subject = st.text_input("Email Subject", 
                                   placeholder="Enter the email subject line")
            body = st.text_area("Email Body", height=200, 
                               placeholder="Paste the full email body text here...")
        
        # File upload for attachments
        uploaded_file = st.file_uploader("Upload Attachment (if any)", 
                                        type=["pdf", "doc", "docx", "xls", "xlsx", "zip", "exe", "txt", "jpg", "png"])
        
        # Additional options
        with st.expander("Advanced Options"):
            urls_detected = bool(re.search(r'https?:\/\/', body))
            urls_present = st.checkbox("Contains URLs in body", 
                                      value=urls_detected)
        
        submit_button = st.form_submit_button("Analyze Email")
    
    # Process the input when the form is submitted
    if submit_button:
        # Check if model is loaded
        if not model_loaded:
            st.error("Model and pipeline files not found. Please ensure the model files are available.")
            st.info("Expected files: 'email_classifier_model.pkl' and 'email_classifier_pipeline.pkl'")
            return
        
        # Format the date to match the expected format
        date_str = f"{date_input.strftime('%a, %d %b %Y')} {time_input.strftime('%H:%M:%S')} +0000"
        
        # Process the input data
        input_data = process_input(sender, date_str, subject, body, int(urls_present))
        
        # Manual analysis of email content
        suspicious_indicators = analyze_email_text(body)
        
        # Process attachment with VirusTotal if present
        attachment_report = None
        if uploaded_file is not None:
            if st.session_state.virustotal_api_key:
                st.subheader("Attachment Analysis")
                
                # Read the file content
                file_bytes = uploaded_file.getvalue()
                
                # Process with VirusTotal
                analysis_id = upload_file_to_virustotal(file_bytes, st.session_state.virustotal_api_key)
                
                if analysis_id:
                    attachment_report = get_virustotal_analysis(analysis_id, st.session_state.virustotal_api_key)
                    
                    if attachment_report:
                        display_virustotal_results(attachment_report)
                        
                        # Add attachment risk to indicators
                        stats = attachment_report['data']['attributes']['stats']
                        if stats['malicious'] > 0:
                            suspicious_indicators.append(f"Attachment flagged as malicious by {stats['malicious']} scanners")
                        elif stats['suspicious'] > 0:
                            suspicious_indicators.append(f"Attachment flagged as suspicious by {stats['suspicious']} scanners")
            else:
                st.warning("⚠️ VirusTotal API key not provided. Attachment scanning skipped.")
                st.info("To enable attachment scanning, enter your VirusTotal API key in the sidebar.")
        
        # Show a spinner while processing email content
        with st.spinner('Analyzing email content...'):
            try:
                # Process with pipeline and make prediction
                processed_data = pipeline.transform(input_data)
                prediction = model.predict(processed_data)[0]
                
                # Get confidence scores
                prediction_proba = model.predict_proba(processed_data)[0]
                confidence = prediction_proba[1] * 100 if prediction == 1 else prediction_proba[0] * 100
                
                # Adjust prediction based on suspicious indicators and attachment analysis
                if len(suspicious_indicators) >= 3 and prediction == 0:
                    st.warning("⚠️ Manual analysis detected multiple suspicious indicators despite model prediction.")
                    
            except Exception as e:
                st.error(f"Error during prediction: {e}")
                st.info("Please check that the input data format matches what the model expects.")
                return
        
        # Display the results
        st.header("Analysis Results")
        
        # Display prediction result explicitly
        prediction_label = get_prediction_label(prediction)
        
        # Create a container for the prediction result
        result_container = st.container()
        
        # Create a progress bar for visualization (ensure it's a float between 0 and 1)
        confidence_percentage = min(confidence, 100)
        confidence_display = float(confidence_percentage) / 100.0  # Explicitly convert to float
        
        # Display prediction with appropriate styling based on result
        if prediction == 1:
            with result_container:
                st.error(f"📊 Prediction: {prediction_label}")
                st.error(format_prediction(prediction, confidence))
                # Use a fixed value if there's still an issue
                try:
                    st.progress(confidence_display)
                except:
                    # Fallback to a fixed value if conversion fails
                    st.progress(0.75 if confidence > 75 else 0.5)
                st.warning("⚠️ This email shows characteristics of a phishing attempt. Be cautious!")
        else:
            with result_container:
                st.success(f"📊 Prediction: {prediction_label}")
                st.success(format_prediction(prediction, confidence))
                # Use a fixed value if there's still an issue
                try:
                    st.progress(confidence_display)
                except:
                    # Fallback to a fixed value if conversion fails
                    st.progress(0.75 if confidence > 75 else 0.5)
                if len(suspicious_indicators) >= 2:
                    st.warning("⚠️ While the model indicates legitimacy, some suspicious elements were detected.")
                else:
                    st.info("ℹ️ Always be careful with emails from unknown sources.")
        
        # Display email details with analysis
        with st.expander("Email Analysis", expanded=True):
            display_email_details(sender, date_str, subject, body, suspicious_indicators)
        
        # Overall risk assessment
        st.subheader("🔍 Overall Risk Assessment")
        
        # Calculate overall risk score (simple weighted algorithm)
        content_risk = confidence if prediction == 1 else (100 - confidence)
        indicator_risk = min(len(suspicious_indicators) * 20, 100)
        
        # Add attachment risk if available
        attachment_risk = 0
        if attachment_report and 'stats' in attachment_report['data']['attributes']:
            stats = attachment_report['data']['attributes']['stats']
            total_scanners = sum(stats.values())
            if total_scanners > 0:
                attachment_risk = (stats.get('malicious', 0) * 100 + stats.get('suspicious', 0) * 50) / total_scanners
        
        # Calculate weighted average risk
        weights = [0.6, 0.3, 0.1 if attachment_report else 0]  # Adjust weights for content, indicators, attachment
        risk_components = [content_risk, indicator_risk, attachment_risk]
        overall_risk = sum(r * w for r, w in zip(risk_components, weights)) / sum(weights)
        
        # Display risk assessment
        risk_level = "High" if overall_risk > 70 else "Medium" if overall_risk > 30 else "Low"
        risk_color = "red" if risk_level == "High" else "orange" if risk_level == "Medium" else "green"
        
        st.markdown(f"<h3 style='color: {risk_color}'>Overall Risk: {risk_level} ({overall_risk:.1f}%)</h3>", unsafe_allow_html=True)
        
        # Show risk components
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Content Risk", f"{content_risk:.1f}%")
        with col2:
            st.metric("Indicator Risk", f"{indicator_risk:.1f}%")
        if attachment_report:
            with col3:
                st.metric("Attachment Risk", f"{attachment_risk:.1f}%")
        
        # Show additional insights and recommendations
        with st.expander("Phishing Detection Guidelines"):
            st.subheader("Key factors in phishing emails:")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("- **Suspicious sender domains** - Legitimate-looking but slightly modified")
                st.markdown("- **Urgent language** - Creating a sense of urgency or fear")
                st.markdown("- **Poor grammar/spelling** - Professional organizations have editorial standards")
            
            with col2:
                st.markdown("- **Suspicious links** - Hover over links to see where they actually lead")
                st.markdown("- **Requests for personal information** - Legitimate organizations rarely ask for sensitive info via email")
                st.markdown("- **Unexpected attachments** - Be wary of unexpected files, especially executables")
            
            st.subheader("What to do if you suspect phishing:")
            st.markdown("1. **Don't click links or download attachments**")
            st.markdown("2. **Don't reply with personal information**")
            st.markdown("3. **Verify through official channels** - Contact the organization directly through their official website")
            st.markdown("4. **Report the email** to your IT department or email provider")
    
    # Sidebar information
    st.sidebar.title("About")
    st.sidebar.info(
        "This tool uses machine learning to detect potential phishing emails. "
        "It analyzes email attributes and uses VirusTotal API to scan attachments."
    )
    
    st.sidebar.title("Model Information")
    if model_loaded:
        st.sidebar.success("✅ Model and pipeline loaded successfully")
    else:
        st.sidebar.error("❌ Model and pipeline not loaded")
    
    st.sidebar.title("VirusTotal Integration")
    if st.session_state.virustotal_api_key:
        st.sidebar.success("✅ VirusTotal API key configured")
    else:
        st.sidebar.warning("⚠️ VirusTotal API key not provided")
        st.sidebar.info(
            "To scan attachments, please enter your VirusTotal API key. "
            "You can get a free API key at [virustotal.com](https://www.virustotal.com/)"
        )
    
    st.sidebar.title("Tips")
    st.sidebar.warning(
        "Remember that no detection system is 100% accurate. "
        "Always exercise caution with unexpected emails."
    )
    
    # Footer
    st.markdown("---")
    st.caption("Advanced Phishing Email Detector v3.0 | Built with Streamlit, Machine Learning, and VirusTotal Integration")


if __name__ == "__main__":
    main()