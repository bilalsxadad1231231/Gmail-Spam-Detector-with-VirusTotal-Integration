import requests
import time

API_KEY = '4f29dd714a194042ba2c2921bdbe61183fa871159efa6130cc6a4f8405bc9b11'
FILE_PATH = 'CEAS_08.csv'  # Replace with the path to your PDF file

def upload_file(file_path):
    url = 'https://www.virustotal.com/api/v3/files'
    headers = {
        'x-apikey': API_KEY
    }
    with open(file_path, 'rb') as f:
        files = {'file': (file_path, f)}
        response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json()['data']['id']
    else:
        print(f"Error uploading file: {response.status_code}")
        print(response.text)
        return None

def get_analysis_report(analysis_id):
    url = f'https://www.virustotal.com/api/v3/analyses/{analysis_id}'
    headers = {
        'x-apikey': API_KEY
    }
    while True:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            analysis = response.json()
            status = analysis['data']['attributes']['status']
            if status == 'completed':
                return analysis
            else:
                print("Analysis in progress, waiting...")
                time.sleep(15)
        else:
            print(f"Error retrieving analysis report: {response.status_code}")
            print(response.text)
            return None

def main():
    analysis_id = upload_file(FILE_PATH)
    if analysis_id:
        report = get_analysis_report(analysis_id)
        if report:
            stats = report['data']['attributes']['stats']
            print(f"Malicious: {stats['malicious']}")
            print(f"Suspicious: {stats['suspicious']}")
            print(f"Undetected: {stats['undetected']}")
            print(f"Timeout: {stats['timeout']}")
            print(f"Harmless: {stats['harmless']}")
        else:
            print("Failed to retrieve analysis report.")
    else:
        print("File upload failed.")

if __name__ == '__main__':
    main()
