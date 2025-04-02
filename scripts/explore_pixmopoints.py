import os
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import datasets
from tqdm import tqdm
import hashlib
import mimetypes
import shutil

def get_image_path(url):
    """Get the path to the cached image file."""
    data_dir = os.environ.get('MOLMO_DATA_DIR')
    if not data_dir:
        raise ValueError("MOLMO_DATA_DIR environment variable not set")
    
    # The image filename is a hash of the URL (this is how Molmo stores them)
    filename = hashlib.sha256(url.encode()).hexdigest()
    image_path = Path(data_dir) / "torch_datasets" / "pixmo_images" / filename
    return image_path

def get_file_info(filepath):
    """Get information about a downloaded file."""
    if not filepath.exists():
        return None
    
    size = filepath.stat().st_size
    mime_type, _ = mimetypes.guess_type(str(filepath))
    
    # Try to determine the actual image format
    try:
        with Image.open(filepath) as img:
            format = img.format.lower()
    except:
        format = "unknown"
    
    return {
        "size": size,
        "mime_type": mime_type,
        "filename": filepath.name,
        "format": format
    }

def create_viewable_copy(filepath, output_dir):
    """Create a copy of the image with proper extension for easier viewing."""
    if not filepath.exists():
        return None
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the image format
    try:
        with Image.open(filepath) as img:
            format = img.format.lower()
    except:
        format = "jpg"  # default to jpg if we can't determine format
    
    # Create new filename with extension
    new_filename = f"{filepath.stem}.{format}"
    new_path = output_dir / new_filename
    
    # Copy the file
    shutil.copy2(filepath, new_path)
    return new_path

def get_image_dir():
    """Get the directory containing downloaded images."""
    data_dir = os.environ.get('MOLMO_DATA_DIR')
    if not data_dir:
        raise ValueError("MOLMO_DATA_DIR environment variable not set")
    return Path(data_dir) / "torch_datasets" / "pixmo_images"

def get_downloaded_files():
    """Get list of downloaded image files."""
    image_dir = get_image_dir()
    if not image_dir.exists():
        return []
    
    # Get all files in the directory
    files = list(image_dir.glob("*"))
    # Filter out any non-files (directories, etc.)
    files = [f for f in files if f.is_file()]
    return files

def load_dataset():
    """Load the PixMo-Points dataset."""
    print("Loading dataset from HuggingFace...")
    ds = datasets.load_dataset("allenai/pixmo-points", split="train")
    print(f"Dataset loaded with {len(ds)} examples")
    return ds

def visualize_example(image_path, dataset):
    """Visualize an image with its points."""
    try:
        file_info = get_file_info(image_path)
        
        if file_info:
            print(f"\nFile information:")
            print(f"  Filename (SHA256 hash): {file_info['filename']}")
            print(f"  Size: {file_info['size'] / 1024:.1f} KB")
            print(f"  MIME type: {file_info['mime_type']}")
            print(f"  Image format: {file_info['format']}")
            
            # Create a viewable copy
            data_dir = os.environ.get('MOLMO_DATA_DIR')
            viewable_dir = Path(data_dir) / "viewable_images"
            viewable_path = create_viewable_copy(image_path, viewable_dir)
            if viewable_path:
                print(f"  Viewable copy: {viewable_path}")
        
        img = Image.open(image_path)
        
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
        # Try to find matching example in dataset
        for example in dataset:
            if get_image_path(example["image_url"]) == image_path:
                if "points" in example:
                    points = example["points"]
                    for point in points:
                        plt.plot(point["x"], point["y"], 'ro', markersize=10)
                plt.title(f"Label: {example['label']}\nCollection method: {example['collection_method']}")
                break
        
        plt.axis('off')
        plt.show()
        return True
    except Exception as e:
        print(f"Warning: {str(e)}")
        return False

def main():
    try:
        # Get downloaded files
        print("Checking for downloaded images...")
        downloaded_files = get_downloaded_files()
        print(f"Found {len(downloaded_files)} downloaded images")
        
        if not downloaded_files:
            print("\nNo images were found in the cache. Please wait for more images to download.")
            print("You can run the download script with:")
            print("  python scripts/download_data.py pixmopoints")
            return
        
        # Load dataset for metadata
        ds = load_dataset()
        
        # Set random seed for reproducibility
        random.seed(42)
        
        # Sample from downloaded images
        shown_examples = 0
        max_examples = min(3, len(downloaded_files))
        
        while shown_examples < max_examples:
            image_path = random.choice(downloaded_files)
            
            print(f"\nExample {shown_examples + 1}:")
            if visualize_example(image_path, ds):
                shown_examples += 1
    
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please make sure you have internet connection and the dataset is accessible.")

if __name__ == "__main__":
    main()