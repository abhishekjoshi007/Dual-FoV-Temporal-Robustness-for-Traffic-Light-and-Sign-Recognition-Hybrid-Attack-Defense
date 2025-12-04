Quick Usage Examples
Example 1: Load Single Frame
python# Initialize loader
loader = DualFoVDatasetLoader(root_dir='./dataset')

# Load a single frame with all data
frame_data = loader.load_complete_frame(
    odd='highway',
    sequence='20231006-114522-00.15.00-00.15.15@Sogun',
    frame_id='0000001'
)

# Access images
mid_range_img = frame_data['images']['mid_range']
long_range_img = frame_data['images']['long_range']

# Access annotations
traffic_lights = frame_data['annotations']['traffic_lights']
traffic_signs = frame_data['annotations']['traffic_signs']

print(f"Traffic lights detected: {len(traffic_lights.get('objects', []))}")
print(f"Traffic signs detected: {len(traffic_signs.get('objects', []))}")
Example 2: Iterate Through Sequence
pythonloader = DualFoVDatasetLoader(root_dir='./dataset')

# Iterate through all frames in a sequence
for frame_data in loader.iterate_sequence('highway', '20231006-114522-00.15.00-00.15.15@Sogun'):
    print(f"Processing frame {frame_data['frame_id']}")
    
    # Process images
    mid_img = frame_data['images']['mid_range']
    
    # Process annotations
    for obj in frame_data['annotations']['traffic_lights'].get('objects', []):
        print(f"  Light {obj['id']}: {obj['state']} at {obj['distance']:.1f}m")
Example 3: Load All Sequences from an ODD
pythonloader = DualFoVDatasetLoader(root_dir='./dataset')

# Get all highway sequences
highway_sequences = loader.sequences['highway']

for seq in highway_sequences:
    print(f"Processing sequence: {seq}")
    
    # Get frame count
    frame_ids = loader.get_frame_ids('highway', seq)
    print(f"  Total frames: {len(frame_ids)}")
    
    # Load calibration (once per sequence)
    calib = loader.load_calibration('highway', seq)
    print(f"  Cameras: {list(calib.keys())}")