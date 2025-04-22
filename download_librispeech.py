
import os
import tarfile
import requests
from tqdm import tqdm

def download_file(url, save_path):
    """
    Download a file from a URL with progress bar
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Check if file already exists
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}")
        return save_path
    
    # Download the file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for HTTP errors
    
    # Get file size for progress bar
    total_size = int(response.headers.get('content-length', 0))
    
    # Download with progress bar
    with open(save_path, 'wb') as file, tqdm(
        desc=os.path.basename(save_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = file.write(chunk)
            bar.update(size)
    
    return save_path

def extract_tarfile(tar_path, extract_path):
    """
    Extract a tar.gz file
    """
    os.makedirs(extract_path, exist_ok=True)
    
    with tarfile.open(tar_path) as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="Extracting"):
            tar.extract(member, path=extract_path)
    
    return extract_path

DATA_CONFIG = {
    "test-clean": {
        "download_mirror": "https://us.openslr.org/resources/12/test-clean.tar.gz",
    },
    "train-clean-360": {
        "download_mirror": "https://us.openslr.org/resources/12/train-clean-360.tar.gz",
    }
}

if __name__ == "__main__":
    # argparse for dataset_name
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True)
    args = parser.parse_args()

    dataset_name = args.dataset_name
    download_mirror = DATA_CONFIG[dataset_name]["download_mirror"]
    data_dir = "data"
    tar_path = os.path.join(data_dir, f"{dataset_name}.tar.gz")
    extract_path = os.path.join(data_dir, dataset_name)

    print(f"Downloading LibriSpeech {dataset_name} from {download_mirror}...")
    download_file(download_mirror, tar_path)

    print(f"Extracting to {extract_path}...")
    extract_tarfile(tar_path, extract_path)

    print(f"Download and extraction complete for {dataset_name}!")
