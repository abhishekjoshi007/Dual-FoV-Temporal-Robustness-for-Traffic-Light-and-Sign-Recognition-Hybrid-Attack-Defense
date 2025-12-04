import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_frame(frame_data: Dict, camera: str = 'mid_range', 
                   figsize=(15, 10)):
    """
    Visualize a frame with bounding boxes
    
    Args:
        frame_data: Output from load_complete_frame()
        camera: 'mid_range' or 'long_range'
        figsize: Figure size for matplotlib
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # Show image
    img = frame_data['images'][camera]
    ax.imshow(img)
    
    # Draw traffic lights (red boxes)
    for obj in frame_data['annotations']['traffic_lights'].get('objects', []):
        if 'bbox_2d' in obj:  # If 2D bbox is available
            x, y, w, h = obj['bbox_2d']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, f"{obj['state']}", color='red', 
                   fontsize=10, weight='bold')
    
    # Draw traffic signs (blue boxes)
    for obj in frame_data['annotations']['traffic_signs'].get('objects', []):
        if 'bbox_2d' in obj:
            x, y, w, h = obj['bbox_2d']
            rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                    edgecolor='blue', facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y-5, obj.get('text', obj['type']), color='blue', 
                   fontsize=10, weight='bold')
    
    ax.axis('off')
    ax.set_title(f"{frame_data['odd'].upper()} - {camera.replace('_', ' ').title()} - Frame {frame_data['frame_id']}")
    plt.tight_layout()
    plt.show()

# Usage
frame_data = loader.load_complete_frame('highway', '20231006-114522-00.15.00-00.15.15@Sogun', '0000001')
visualize_frame(frame_data, camera='mid_range')