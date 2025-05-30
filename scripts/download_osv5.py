#!/usr/bin/env python3
"""
OSV5M Dataset Download Script
Downloads the OSV5M dataset excluding train images, extracts to specified directories.
"""

import os
import zipfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import argparse


def create_directories():
    """Create necessary directory structure."""
    directories = [
        "data/downloads/osv5",
        "data/osv5/images"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def download_test_images(repo_id="osv5m/osv5m", download_dir="data/downloads/osv5"):
    """Download test image zip files."""
    print("Downloading test images...")
    
    try:
        # List all files in the images/test folder
        repo_files = list_repo_files(repo_id=repo_id, repo_type='dataset')
        test_zip_files = [f for f in repo_files if f.startswith("images/test/") and f.endswith(".zip")]
        
        downloaded_files = []
        for zip_file in test_zip_files:
            filename = os.path.basename(zip_file)
            print(f"Downloading {zip_file}...")
            
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=zip_file,
                repo_type='dataset',
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            downloaded_files.append(local_path)
            
        print(f"Downloaded {len(downloaded_files)} test image zip files")
        return downloaded_files
        
    except Exception as e:
        print(f"Error downloading test images: {e}")
        return []


def download_csv_files(repo_id="osv5m/osv5m", download_dir="data/downloads/osv5"):
    """Download CSV files."""
    print("Downloading CSV files...")
    
    csv_files = ["test.csv", "train.csv"]  # Download both, we'll only move test.csv later
    downloaded_csvs = []
    
    for csv_file in csv_files:
        try:
            print(f"Downloading {csv_file}...")
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=csv_file,
                repo_type='dataset',
                local_dir=download_dir,
                local_dir_use_symlinks=False
            )
            downloaded_csvs.append(local_path)
        except Exception as e:
            print(f"Warning: Could not download {csv_file}: {e}")
    
    return downloaded_csvs


def extract_zip_files(download_dir="data/downloads/osv5", extract_dir="data/osv5/images"):
    """Extract all zip files to the target directory."""
    print("Extracting zip files...")
    
    zip_files = []
    for root, dirs, files in os.walk(download_dir):
        for file in files:
            if file.endswith(".zip"):
                zip_files.append(os.path.join(root, file))
    
    extracted_count = 0
    for zip_path in zip_files:
        try:
            print(f"Extracting {os.path.basename(zip_path)}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
            extracted_count += 1
            
            # Remove the zip file after extraction
            os.remove(zip_path)
            print(f"Removed {os.path.basename(zip_path)}")
            
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
    
    print(f"Extracted {extracted_count} zip files")


def move_csv_files(download_dir="data/downloads/osv5", target_dir="data/osv5"):
    """Move test.csv to the target directory."""
    print("Moving CSV files...")
    
    test_csv_source = os.path.join(download_dir, "test.csv")
    test_csv_target = os.path.join(target_dir, "test.csv")
    
    if os.path.exists(test_csv_source):
        # Ensure target directory exists
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        
        shutil.move(test_csv_source, test_csv_target)
        print(f"Moved test.csv to {test_csv_target}")
    else:
        print("Warning: test.csv not found in downloads directory")
    
    # Clean up train.csv if it was downloaded
    train_csv_source = os.path.join(download_dir, "train.csv")
    if os.path.exists(train_csv_source):
        os.remove(train_csv_source)
        print("Removed train.csv (not needed)")


def cleanup_downloads(download_dir="data/downloads"):
    """Clean up the downloads directory."""
    print("Cleaning up downloads directory...")
    
    # Check if directory is empty
    if os.path.exists(download_dir):
        remaining_files = os.listdir(download_dir)
        if not remaining_files:
            os.rmdir(download_dir)
            print("Removed empty downloads directory")
        else:
            print(f"Downloads directory contains {len(remaining_files)} remaining files")


def main():
    """Main function to orchestrate the download process."""
    parser = argparse.ArgumentParser(description="Download OSV5M dataset (excluding train images)")
    parser.add_argument("--keep-downloads", action="store_true", 
                       help="Keep the downloads directory after extraction")
    args = parser.parse_args()
    
    print("Starting OSV5M dataset download (excluding train images)...")
    print("=" * 60)
    
    # Create directory structure
    create_directories()
    
    # Download test images
    downloaded_zips = download_test_images()
    
    # Download CSV files
    downloaded_csvs = download_csv_files()
    
    if not downloaded_zips and not downloaded_csvs:
        print("No files downloaded. Exiting.")
        return
    
    # Extract zip files
    extract_zip_files()
    
    # Move CSV files to target locations
    move_csv_files()
    
    # Cleanup downloads directory unless requested to keep
    if not args.keep_downloads:
        cleanup_downloads()
    
    print("=" * 60)
    print("Download and extraction complete!")
    print(f"Test images extracted to: data/osv5/images/")
    print(f"Test CSV file saved to: data/osv5/test.csv")
    
    # Display final directory structure
    print("\nFinal directory structure:")
    for root, dirs, files in os.walk("data"):
        level = root.replace("data", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files[:5]:  # Show first 5 files
            print(f"{subindent}{file}")
        if len(files) > 5:
            print(f"{subindent}... and {len(files) - 5} more files")


if __name__ == "__main__":
    main()