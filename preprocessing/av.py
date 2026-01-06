import pandas as pd
from PIL import Image, UnidentifiedImageError
import io
import os
import numpy as np
import base64
import re

def safe_filename(s):
    return re.sub(r'[^a-zA-Z0-9._]', '_', str(s))

def get_image_bytes(img_data):
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

# List of input parquet files
input_files = [
    r"C:\Users\rohan\Downloads\testing_camera_image_10980133015080705026_780_000_800_000.parquet",
    r"C:\Users\rohan\Downloads\testing_camera_image_11987368976578218644_1340_000_1360_000.parquet",
    r"C:\Users\rohan\Downloads\testing_camera_image_14737335824319407706_1980_000_2000_000.parquet",
    r"C:\Users\rohan\Downloads\testing_camera_image_14188689528137485670_2660_000_2680_000.parquet",
    r"C:\Users\rohan\Downloads\testing_camera_image_17136775999940024630_4860_000_4880_000.parquet",
    r"C:\Users\rohan\Downloads\testing_camera_image_15272375112495403395_620_000_640_000.parquet"
]

output_base_dir = r"C:\Users\rohan\OneDrive\Desktop\Genai\AVML\training_camera_image"

camera_id = 1  # FRONT camera only
camera_name = "front"

image_column = '[CameraImageComponent].image'

error_count = 0

for idx, parquet_file in enumerate(input_files, start=1):
    print(f"\nProcessing file {idx}/{len(input_files)}: {parquet_file}")

    # Load parquet file
    df = pd.read_parquet(parquet_file)
    df = df[df['key.camera_name'] == camera_id].copy()
    df = df.sort_values(['key.frame_timestamp_micros']).reset_index(drop=True)

    # Create folder front1, front2, ...
    folder_name = f"{camera_name}{idx}"
    output_dir = os.path.join(output_base_dir, folder_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving FRONT camera images to folder: {folder_name} ({len(df)} frames)")

    for seq_idx, (index, row) in enumerate(df.iterrows()):
        try:
            timestamp = row['key.frame_timestamp_micros']
            img_data = row[image_column]

            img_filename = f"{camera_name}_frame_{seq_idx:04d}_{timestamp}.jpg"

            img_bytes = get_image_bytes(img_data)
            if img_bytes is not None:
                try:
                    img = Image.open(io.BytesIO(img_bytes))
                    img.load()
                    img.save(os.path.join(output_dir, img_filename))
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Error saving frame {seq_idx}: {e}")
                    error_count += 1
            else:
                print(f"Frame {seq_idx}: Unsupported image format")
                error_count += 1

            if seq_idx % 20 == 0:
                print(f"  Processed {seq_idx + 1} frames...")

        except Exception as e:
            error_count += 1
            print(f"Error processing frame {seq_idx}: {e}")

print(f"\nFinished processing all files with {error_count} errors.")
print(f"Images saved to: {output_base_dir}")
