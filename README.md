# Email Classification Model

This project implements a machine learning-based email classification system that can analyze and classify emails based on their content, sender, subject, and other features. It includes both a Python API and a Streamlit web application with VirusTotal integration for enhanced security analysis.

## Key Features

- **Gmail Integration**:
  - Direct access to Gmail inbox
  - Real-time email monitoring
  - Secure OAuth 2.0 authentication
  - Automatic email fetching and analysis

- **Advanced Analysis Pipeline**:
  - Dual-model classification (XGBoost and BERT)
  - Attachment security scanning via VirusTotal
  - Real-time threat assessment
  - Comprehensive spam detection

- **Security Features**:
  - VirusTotal integration for file analysis
  - Multiple antivirus engine scanning
  - Threat level classification
  - Suspicious pattern detection
  - Secure file handling

- **User Interface**:
  - Interactive Streamlit web interface
  - Real-time analysis results
  - Visual threat indicators
  - Detailed security reports
  - Easy email selection and analysis

## Features

- Single email classification
- Batch email classification
- Pre-trained model support
- Preprocessing pipeline for consistent data handling
- Support for both dictionary and DataFrame input formats
- Interactive Streamlit web interface
- VirusTotal integration for file analysis
- Real-time threat assessment
- Detailed security scan results
- Gmail integration for direct email analysis
- Real-time email monitoring
- Automatic spam detection for incoming emails

## Prerequisites

- Python 3.x
- Required Python packages:
  - pandas
  - scikit-learn
  - xgboost
  - pickle
  - streamlit
  - requests
  - numpy
  - tensorflow
  - transformers
  - torch
  - VirusTotal API key (for file analysis feature)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up VirusTotal API key (optional, for file analysis):
   - Get an API key from [VirusTotal](https://www.virustotal.com/gui/join-us)
   - Set it as an environment variable or in the Streamlit secrets

## Usage

### Web Application

To run the Streamlit web application:

```bash
streamlit run app.py
```

The web application provides:
- Interactive email analysis interface
- File upload and VirusTotal scanning
- Real-time threat assessment
- Detailed security scan results
- Visual indicators for potential threats

### Python API

#### Single Email Classification

```python
from testModel import load_model, predict_single_email

# Load the model and pipeline
model, pipeline = load_model()

# Example email data
test_email = {
    'sender': 'example@domain.com',
    'date': 'Mon, 14 Apr 2025 10:30:45 +0000',
    'subject': 'Example Subject',
    'body': 'Email body content...',
    'urls': None
}

# Make prediction
prediction = predict_single_email(test_email, model, pipeline)
print(f"Prediction: {prediction}")
```

#### Batch Classification

For batch classification, prepare your data in a CSV file with the following columns:
- sender
- date
- subject
- body
- urls

Then use the batch prediction function:

```python
from testModel import load_model, batch_predict
import pandas as pd

# Load the model and pipeline
model, pipeline = load_model()

# Load your CSV file
test_df = pd.read_csv('your_emails.csv')

# Make batch predictions
predictions = batch_predict(test_df, model, pipeline)

# Save results
test_df['predicted_label'] = predictions
test_df.to_csv('emails_with_predictions.csv', index=False)
```

## Model Files

The project uses multiple model files:
- `email_classifier_model.pkl`: The trained XGBoost model
- `email_classifier_pipeline.pkl`: The preprocessing pipeline
- `bert_model/`: Directory containing BERT model files and configurations

### Available Models

1. **XGBoost Model**
   - Traditional machine learning approach
   - Uses TF-IDF features for text processing
   - Handles structured features like date, sender, and URLs
   - Fast inference time
   - Good for general email classification tasks

2. **BERT Model**
   - State-of-the-art transformer-based model
   - Better understanding of contextual information
   - Superior performance on complex text patterns
   - Handles long-form text effectively
   - Requires more computational resources
   - Better for nuanced classification tasks

### Model Selection

The system supports both models and can be configured based on your needs:
- Use XGBoost for faster inference and lower resource requirements
- Use BERT for higher accuracy and better handling of complex text patterns

## Project Structure

```
├── README.md
├── requirements.txt
├── app.py                 # Streamlit web application
├── trainModel.py          # Model training script
├── testModel.py          # Model testing and prediction script
├── email_classifier_model.pkl
├── email_classifier_pipeline.pkl
├── bert_model/           # BERT model directory
│   ├── config.json       # BERT model configuration
│   ├── pytorch_model.bin # BERT model weights
│   └── vocab.txt        # BERT vocabulary file
└── data/                 # Training data directory
    └── *.csv            # Training datasets
```

## Security Features

The application includes several security features:
- VirusTotal integration for file analysis
- Real-time threat assessment
- Multiple antivirus engine scanning
- Threat level classification (High/Medium/Low)
- Detailed security scan results
- Visual indicators for potential threats

## Model Training

The project supports two different approaches for training email classification models:

### 1. XGBoost Model Training

The XGBoost model can be trained using the `trainModel.py` script:

```bash
python trainModel.py
```

This script:
- Handles data preprocessing and feature extraction
- Implements multiple model architectures (XGBoost, Random Forest, SVM)
- Performs cross-validation and model evaluation
- Saves the best performing model and preprocessing pipeline

### 2. BERT Model Training

The BERT model training is implemented in the Jupyter notebook `BERT_spam_classification.ipynb`. This notebook:

1. **Data Preparation**:
   - Downloads and processes the Enron email dataset
   - Handles class imbalance through downsampling
   - Performs data cleaning and preprocessing

2. **Model Training**:
   - Uses the Hugging Face Transformers library
   - Implements BERT-based classification
   - Includes GPU support for faster training
   - Saves the trained model and tokenizer

3. **Training Process**:
   ```python
   # Install required packages
   pip install nltk torch transformers tensorflow-text
   
   # Run the notebook
   jupyter notebook BERT_spam_classification.ipynb
   ```

4. **Model Output**:
   - Saves the trained BERT model in the `bert_model/` directory
   - Includes model configuration, weights, and vocabulary files

### Choosing Between Training Approaches

- **Use XGBoost Training** when:
  - You need faster training times
  - Have limited computational resources
  - Want to include structured features (date, sender, URLs)
  - Need a simpler deployment process

- **Use BERT Training** when:
  - You need higher accuracy
  - Have access to GPU resources
  - Want better handling of complex text patterns
  - Need state-of-the-art performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Add your license information here]

## Contact

[Add your contact information here]

## Gmail Integration

The application includes Gmail API integration for direct email analysis and monitoring. This feature allows you to:

1. **Connect Your Gmail Account**:
   - Set up OAuth 2.0 credentials in Google Cloud Console
   - Enable Gmail API access
   - Authenticate using the provided credentials

2. **Email Monitoring**:
   - Real-time analysis of incoming emails
   - Automatic spam detection
   - Batch processing of existing emails
   - Detailed analysis reports

3. **Setup Instructions**:
   ```bash
   # 1. Create a project in Google Cloud Console
   # 2. Enable Gmail API
   # 3. Create OAuth credentials
   # 4. Download credentials.json to your project directory
   ```

4. **Required Files**:
   - `cred.json`: Your Gmail API credentials
   - `token.pickle`: Generated after first authentication
   - `emailSendTest.py`: Gmail integration script

5. **Usage**:
   ```python
   # Check new emails
   python emailSendTest.py
   
   # Or use the Streamlit interface
   streamlit run app.py
   ```

6. **Security Features**:
   - Secure OAuth 2.0 authentication
   - Read-only access to emails
   - No email content storage
   - Automatic token refresh 

## Application Workflow

The application follows a comprehensive workflow for email analysis and security assessment:

1. **Gmail Authentication and Email Fetching**:
   ```mermaid
   graph TD
   A[Start] --> B[Load cred.json]
   B --> C[Authenticate with Gmail API]
   C --> D[Fetch Recent Emails]
   D --> E[Display Email List in Streamlit]
   ```

2. **Email Analysis Pipeline**:
   ```mermaid
   graph TD
   A[Select Email] --> B[Extract Email Content]
   B --> C{Contains Attachments?}
   C -->|Yes| D[Download Attachments]
   C -->|No| E[Text Analysis]
   D --> F[VirusTotal Scan]
   F --> G[Security Assessment]
   E --> H[Spam Classification]
   H --> I[Display Results]
   G --> I
   ```

3. **Detailed Process Flow**:

   a. **Initial Setup**:
   - Application loads Gmail credentials from `cred.json`
   - Authenticates with Gmail API
   - Creates a secure session for email access

   b. **Email Listing**:
   - Fetches recent emails from Gmail inbox
   - Displays email list in Streamlit interface
   - Shows sender, subject, and date for each email

   c. **Email Selection and Analysis**:
   - User selects an email for analysis
   - System extracts email content and metadata
   - Performs initial spam classification

   d. **Attachment Processing**:
   - If email contains attachments:
     1. Downloads attachments securely
     2. Uploads to VirusTotal for scanning
     3. Waits for scan results
     4. Analyzes security threats
     5. Generates threat assessment report

   e. **Content Analysis**:
   - Analyzes email body text
   - Checks for suspicious patterns
   - Identifies potential phishing attempts
   - Performs spam classification using both:
     - XGBoost model
     - BERT model (if configured)

   f. **Results Display**:
   - Shows comprehensive analysis results:
     - Spam probability
     - Threat level assessment
     - VirusTotal scan results (if applicable)
     - Suspicious content indicators
     - Recommended actions

4. **Security Features**:
   - Secure OAuth 2.0 authentication
   - Read-only email access
   - Temporary file handling
   - Automatic token refresh
   - Secure API key management

5. **Example Usage**:
   ```python
   # 1. Start the application
   streamlit run app.py

   # 2. Authenticate with Gmail
   # - Enter credentials when prompted
   # - Grant necessary permissions

   # 3. View and Analyze Emails
   # - Browse through email list
   # - Select email for analysis
   # - View detailed results
   # - Check attachment security
   ```

6. **Output Example**:
   ```
   Email Analysis Results:
   ----------------------
   Sender: example@domain.com
   Subject: Important Document
   Date: 2024-04-24 10:30:00
   
   Spam Classification:
   - XGBoost: 95% Spam
   - BERT: 92% Spam
   
   Threat Assessment:
   - Level: High
   - Suspicious Patterns: 3
   - Malicious Links: 1
   
   Attachment Analysis:
   - document.pdf: Clean
   - spreadsheet.xlsx: Suspicious
   
   Recommended Action: Quarantine
   ``` 