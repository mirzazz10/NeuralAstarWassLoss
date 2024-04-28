"""
Written By Rahman Baig Mirza

"""

import os
import urllib.request
import zipfile

# Create directories
directories = [
    "data/street/original",
    "data/street/original/all",
    "data/street/original/mixed/train",
    "data/street/original/mixed/validation",
    "data/street/original/mixed/test"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Download the file
url = "https://www.movingai.com/benchmarks/street/street-png.zip"
download_path = "data/street/original/street-png.zip"
urllib.request.urlretrieve(url, download_path)

# Unzip the file
zip_file_path = "data/street/original/street-png.zip"
extract_to_path = "data/street/original/all/"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
