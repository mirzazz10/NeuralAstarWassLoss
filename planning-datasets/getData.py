# import urllib
# import urllib.request
# import os
# def download_url(url, root, filename=None):
#     """Download a file from a url and place it in root.
#     Args:
#         url (str): URL to download file from
#         root (str): Directory to place downloaded file in
#         filename (str, optional): Name to save the file under. If None, use the basename of the URL
#     """

#     # root = os.path.expanduser(root)
#     # if not filename:
#     #     filename = os.path.basename(url)
#     # fpath = os.path.join(root, filename)

#     # os.makedirs(root, exist_ok=True)

#     try:
#         print('Downloading ' + url + ' to ' + fpath)
#         urllib.request.urlretrieve(url, fpath)
#     except (urllib.error.URLError, IOError) as e:
#         if url[:5] == 'https':
#             url = url.replace('https:', 'http:')
#             print('Failed download. Trying https -> http instead.'
#                     ' Downloading ' + url + ' to ' + fpath)
#             urllib.request.urlretrieve(url, fpath)

# if __name__ == "__main__":
#     url = "https://www.movingai.com/benchmarks/street/street-png.zip" 
#     fpath = "data/street/original/street-png.zip"
#     print( "Start")
#     download_url( url, fpath)

#     # unzip the data


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
