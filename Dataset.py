import os
import subprocess
import zipfile
import json

# Ensure kaggle API key is in place
kaggle_api_dir = os.path.expanduser('~/.kaggle')
os.makedirs(kaggle_api_dir, exist_ok=True)

# Write your Kaggle API key to a file
kaggle_api_key = {
    "username": "YOUR_USERNAME",
    "key": "YOUR_API_KEY"
}

with open(os.path.join(kaggle_api_dir, '../kaggle.json'), 'w') as file:
    json.dump(kaggle_api_key, file)

# Set file permissions
os.chmod(os.path.join(kaggle_api_dir, '../kaggle.json'), 0o600)

# Download the dataset from Kaggle using subprocess
subprocess.run(['kaggle', 'datasets', 'download', '-d', 'robinreni/revitsone-5class', '--force'], check=True)

# Unzip the dataset using zipfile module
with zipfile.ZipFile('../revitsone-5class.zip', 'r') as zip_ref:
    zip_ref.extractall('revitsone-5class')

print("Dataset downloaded and extracted successfully.")
