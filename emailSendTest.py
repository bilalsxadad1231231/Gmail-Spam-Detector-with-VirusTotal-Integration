import os
import pickle
import base64
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from email.mime.text import MIMEText
import datetime

# Define the required scopes
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """
    Authenticates the user and returns the Gmail service object.
    The authentication token is saved to 'token.pickle' for future use.
    """
    creds = None
    
    # Try to load existing credentials from token.pickle
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    
    # If credentials don't exist or are invalid, authenticate
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            print("No valid credentials found. Please authenticate.")
            flow = InstalledAppFlow.from_client_secrets_file(
                'cred.json', SCOPES)
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    
    # Build the Gmail service
    return build('gmail', 'v1', credentials=creds)

def list_new_emails(count=10):
    """
    Lists the specified number of latest emails from the user's Gmail inbox.
    
    Args:
        count: Number of emails to retrieve (default 10)
    """
    service = get_gmail_service()
    
    # Get emails
    results = service.users().messages().list(
        userId='me', 
        labelIds=['INBOX'],
        maxResults=count
    ).execute()
    
    messages = results.get('messages', [])
    
    if not messages:
        print("No new messages found.")
        return
    
    print(f"Found {len(messages)} new messages:\n")
    
    # Process each message
    for message in messages:
        msg = service.users().messages().get(
            userId='me', 
            id=message['id'],
            format='metadata',
            metadataHeaders=['From', 'Subject', 'Date']
        ).execute()
        
        headers = msg['payload']['headers']
        
        # Extract email details
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), 'No Subject')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Unknown Sender')
        date = next((h['value'] for h in headers if h['name'] == 'Date'), 'Unknown Date')
        
        # Get the email URL
        email_url = f"https://mail.google.com/mail/u/0/#inbox/{message['id']}"
        
        # Print email info
        print(f"From: {sender}")
        print(f"Subject: {subject}")
        print(f"Date: {date}")
        print(f"Link: {email_url}")
        print("-" * 50)

if __name__ == "__main__":
    try:
        print("Checking for new emails in Gmail...")
        list_new_emails(20)  # Change the number to show more or fewer emails
    except Exception as e:
        print(f"An error occurred: {e}")
        print("\nMake sure you have set up the Gmail API properly:")
        print("1. Create a project in Google Cloud Console")
        print("2. Enable Gmail API")
        print("3. Create OAuth credentials")
        print("4. Download the credentials.json file to the same directory as this script")