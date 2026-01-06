import pandas as pd
from PIL import Image, UnidentifiedImageError
import io
import os
import numpy as np
import base64
import re

def safe_filename(s):
    """Replace any character that is not alphanumeric, dot, or underscore with underscore."""
    return re.sub(r'[^a-zA-Z0-9._]', '_', str(s))

def get_image_bytes(img_data):
    """Convert image data to bytes if possible, else return None."""
    if isinstance(img_data, bytes):
        return img_data
    elif isinstance(img_data, memoryview):
        return img_data.tobytes()
    elif isinstance(img_data, np.ndarray):
        return img_data.tobytes()
    elif isinstance(img_data, str):
        try:
            return base64.b64decode(img_data)
        except Exception:
            try:
                return bytes.fromhex(img_data)
            except Exception:
                return None
    else:
        return None

# === File paths ===
camera_image_file = r"C:\Users\rohan\Downloads\training_camera_image_10017090168044687777_6380_000_6400_000.parquet"
camera_box_file = r"C:\Users\rohan\Downloads\training_camera_box_10017090168044687777_6380_000_6400_000.parquet"
camera_calibration_file = r"C:\Users\rohan\Downloads\training_camera_calibration_10017090168044687777_6380_000_6400_000.parquet"
output_csv_path = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\final_merged_dataset.csv"
output_image_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\training_images_cemeranew"
os.makedirs(output_image_dir, exist_ok=True)

# === Camera name mapping for better organization ===
camera_names = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

print("Reading parquet files...")
camera_image_df = pd.read_parquet(camera_image_file)
camera_box_df = pd.read_parquet(camera_box_file)
camera_calibration_df = pd.read_parquet(camera_calibration_file)

# === Sort ONLY by timestamp to maintain temporal sequence ===
camera_image_df = camera_image_df.sort_values(['key.frame_timestamp_micros']).reset_index(drop=True)
camera_box_df = camera_box_df.sort_values(['key.frame_timestamp_micros']).reset_index(drop=True)

# === Find the image column ===
image_column = '[CameraImageComponent].image'

print("Extracting and saving images organized by camera...")
image_filenames = []
error_count = 0

# === Group by camera to create separate video sequences ===
for camera_id, camera_name in camera_names.items():
    camera_subset = camera_image_df[camera_image_df['key.camera_name'] == camera_id].copy()
    camera_subset = camera_subset.sort_values(['key.frame_timestamp_micros']).reset_index(drop=True)
    
    print(f"\nProcessing {camera_name} camera ({len(camera_subset)} frames)...")
    
    # Create camera-specific directory
    camera_dir = os.path.join(output_image_dir, camera_name)
    os.makedirs(camera_dir, exist_ok=True)
    
    for seq_idx, (index, row) in enumerate(camera_subset.iterrows()):
        try:
            segment_id = safe_filename(row['key.segment_context_name'])
            timestamp = row['key.frame_timestamp_micros']
            img_data = row[image_column]
            
            # Create sequential filename for video-like sequence
            img_filename = f"{camera_name}_frame_{seq_idx:04d}_{timestamp}.jpg"
            
            img_bytes = get_image_bytes(img_data)
            if img_bytes is not None:
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    img.load()
                    img.save(os.path.join(camera_dir, img_filename))
                    
                    # Update the main dataframe with the new filename
                    camera_image_df.loc[index, 'image_filename'] = f"{camera_name}/{img_filename}"
                    
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Error with {camera_name} frame {seq_idx}: {e}")
                    camera_image_df.loc[index, 'image_filename'] = "ERROR"
                    error_count += 1
            else:
                print(f"{camera_name} frame {seq_idx}: Unsupported image format")
                camera_image_df.loc[index, 'image_filename'] = "ERROR"
                error_count += 1
                
            if seq_idx % 20 == 0:
                print(f"  Processed {seq_idx+1} {camera_name} frames...")
                
        except Exception as e:
            error_count += 1
            print(f"Error processing {camera_name} frame {seq_idx}: {e}")
            camera_image_df.loc[index, 'image_filename'] = "ERROR"

print(f"\nFinished saving images with {error_count} errors.")

# === Create a summary file for video-like sequences ===
for camera_id, camera_name in camera_names.items():
    camera_subset = camera_image_df[camera_image_df['key.camera_name'] == camera_id].copy()
    camera_subset = camera_subset.sort_values(['key.frame_timestamp_micros']).reset_index(drop=True)
    
    if len(camera_subset) > 0:
        camera_csv = f"C:\\Users\\rohan\\OneDrive\\Desktop\\Genai\\AVML\\{camera_name}_sequence.csv"
        camera_subset.to_csv(camera_csv, index=False)
        print(f"Saved {camera_name} sequence with {len(camera_subset)} frames to {camera_csv}")

# === Drop the image column and merge with other data ===
camera_image_df = camera_image_df.drop(columns=[image_column])

print("Merging dataframes...")
image_box_keys = ['key.segment_context_name', 'key.frame_timestamp_micros', 'key.camera_name']
calibration_keys = ['key.segment_context_name', 'key.camera_name']

merged_df = pd.merge(camera_image_df, camera_box_df, on=image_box_keys, how='inner')
final_df = pd.merge(merged_df, camera_calibration_df, on=calibration_keys, how='inner')

final_df.to_csv(output_csv_path, index=False)
print(f"\nFinal merged CSV saved to: {output_csv_path}")
print(f"Images organized by camera in: {output_image_dir}")
print("\nFolder structure:")
print(f"  {output_image_dir}/")
for camera_name in camera_names.values():
    print(f"    ├── {camera_name}/")
    print(f"    │   ├── {camera_name}_frame_0000_[timestamp].jpg")
    print(f"    │   ├── {camera_name}_frame_0001_[timestamp].jpg")
    print(f"    │   └── ...")
