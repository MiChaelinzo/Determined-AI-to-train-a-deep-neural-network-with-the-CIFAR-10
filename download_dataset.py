import requests
import tarfile
import os

url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
filename = "cifar-10-python.tar.gz"
extracted_folder_name = "cifar-10-batches-py"

# Create a data directory if it doesn't exist
if not os.path.exists("data"):
    os.makedirs("data")

# Download the dataset
print("Downloading CIFAR-10 dataset...")
response = requests.get(url, stream=True)
with open(f"data/{filename}", "wb") as f:
    for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)

# Extract the dataset
print("Extracting CIFAR-10 dataset...")
with tarfile.open(f"data/{filename}", "r:gz") as tar:
    tar.extractall("data")

print("Done.")
