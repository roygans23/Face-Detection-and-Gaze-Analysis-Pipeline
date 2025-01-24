import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

def load_and_process_data(csv_path, frame_skip):
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for frames that are multiples of frame_skip
    df = df[df['frame'] % frame_skip == 0]
    
    # Convert string representations of lists to actual lists
    df['top_left'] = df['top_left'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    df['bottom_right'] = df['bottom_right'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    
    return df

def is_gaze_in_bbox(gaze_x, gaze_y, top_left, bottom_right):
    # If there's no bounding box (NaN or None), return False
    if top_left is None or bottom_right is None:
        return False
    
    try:
        return (top_left[0] <= gaze_x <= bottom_right[0] and 
                top_left[1] <= gaze_y <= bottom_right[1])
    except:
        return False

def get_actor_images(actors_path):
    actor_images = {}
    for actor in os.listdir(actors_path):
        actor_dir = os.path.join(actors_path, actor)
        if os.path.isdir(actor_dir):
            # Get first image in actor's directory
            image_files = [f for f in os.listdir(actor_dir) if f.endswith(('.jpg', '.png'))]
            if image_files:
                img_path = os.path.join(actor_dir, image_files[0])
                try:
                    img = Image.open(img_path)
                    # Convert to RGB mode if necessary
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    # Resize image to reasonable dimensions
                    img = img.resize((100, 100))
                    actor_images[actor] = img
                except Exception as e:
                    print(f"Error loading image for {actor}: {e}")
    return actor_images

def create_gaze_visualization(df, actors_path, frame_skip):
    # Get actor images silently
    actor_images = get_actor_images(actors_path)
    
    # Create a mapping of normalized actor names (lowercase, no spaces) to actual names
    actor_name_mapping = {name.lower().strip(): name for name in actor_images.keys()}
    
    actors = list(actor_images.keys())
    # Modified color generation for maximum contrast between adjacent actors
    colors = plt.cm.tab20(np.linspace(0, 1, len(actors)))  # Changed to tab20 colormap for more distinct colors
    actor_colors = dict(zip(actors, colors))
    
    # Calculate ratios and store gaze times for each actor
    total_frames = len(df)  # Get total number of frames
    actor_stats = {actor: {'gazes': 0, 'appearances': 0, 'gaze_times': []} for actor in actors}
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        gaze_x, gaze_y = row['Gaze point X'], row['Gaze point Y']
        frame = row['frame']
        
        # Skip if no gaze data
        if pd.isna(gaze_x) or pd.isna(gaze_y):
            continue
            
        # Get the current actor from the row and normalize it
        current_actor = str(row['best_match']).lower().strip()
        
        # Map normalized name back to actual name
        if current_actor in actor_name_mapping:
            actual_actor = actor_name_mapping[current_actor]
            
            # Count appearances (when actor has a valid bounding box)
            if row['top_left'] is not None and row['bottom_right'] is not None:
                actor_stats[actual_actor]['appearances'] += 1
                
                # Check if gaze is in bounding box
                if is_gaze_in_bbox(gaze_x, gaze_y, row['top_left'], row['bottom_right']):
                    actor_stats[actual_actor]['gazes'] += 1
                    actor_stats[actual_actor]['gaze_times'].append(frame)

    # Calculate max_frame before the plotting loop
    max_frame = df['frame'].max()

    # Create visualization with MUCH larger figure size
    height_per_actor = 1  # Height per actor
    fig, ax = plt.subplots(figsize=(60, len(actors) * height_per_actor * 1.5))  # Increased width from 30 to 40
    
    # Set background color
    ax.set_facecolor('white')
    fig.set_facecolor('white')
    
    # Plot for each actor
    for idx, actor in enumerate(actors):
        try:
            # Plot actor image
            img = actor_images[actor]
            img_array = np.array(img)
            
            # Adjust image position and size - made images SUPER wide
            img_height = 0.8  # Height of image
            y_position = idx
            # Modified extent to make images SUPER wide: [left, right, bottom, top]
            extent = [-40.0, -2.0, y_position-img_height/2, y_position+img_height/2]
            
            # Plot image with higher zorder and no transparency
            ax.imshow(img_array, extent=extent, aspect='auto', alpha=1.0, zorder=2)
            
            # Plot vertical lines for gazes - match image height exactly
            y_bottom = (y_position - img_height/2 + 1) / (len(actors) + 1)  # Normalize to [0,1] range
            y_top = (y_position + img_height/2 + 1) / (len(actors) + 1)     # Normalize to [0,1] range
            
            for gaze_time in actor_stats[actor]['gaze_times']:
                ax.axvline(x=gaze_time/frame_skip, 
                          ymin=y_bottom,
                          ymax=y_top, 
                          color=actor_colors[actor], 
                          alpha=0.5,
                          zorder=1)
            
            # Add statistics text on the right side
            appearances = actor_stats[actor]['appearances']
            gazes = actor_stats[actor]['gazes']
            gaze_ratio = (gazes / appearances * 100) if appearances > 0 else 0
            
            # Calculate frame presence ratio
            frame_presence_ratio = (appearances / total_frames * 100)
            
            stats_text = (f"{actor}\n"
                         f"Appearances: {appearances}\n"
                         f"Gaze ratio: {gaze_ratio:.1f}%\n"
                         f"Frame presence: {frame_presence_ratio:.1f}%")
            
            # Use the same y_position as the image for alignment
            ax.text(max_frame/frame_skip * 1.02, y_position, stats_text, 
                   verticalalignment='center',
                   fontsize=10)
            
        except Exception as e:
            print(f"Error plotting actor {actor}: {str(e)}")
    
    # Customize plot
    ax.set_ylim(-1, len(actors))
    
    # Set x-axis ticks to multiples of 60 (frame_skip)
    xticks = np.arange(0, max_frame + 1, frame_skip) / frame_skip
    ax.set_xticks(xticks)
    ax.set_xticklabels([int(x * frame_skip) for x in xticks])
    
    ax.set_xlabel('Frames')
    ax.set_title('Gaze Analysis by Actor')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Remove y-axis ticks and labels since we have images
    ax.set_yticks([])
    
    # Adjust margins to give more space for the statistics text
    plt.subplots_adjust(right=0.80, left=0.65)  # Adjusted right margin to make room for text
    
    return plt

# Usage
if __name__ == "__main__":
    CSV_PATH = "data/csvs_outputs/test_video_with_gazes.csv"
    ACTORS_PATH = "data/actors_ready"
    FRAME_SKIP = 60
    
    df = load_and_process_data(CSV_PATH, FRAME_SKIP)
    plt = create_gaze_visualization(df, ACTORS_PATH, FRAME_SKIP)
    plt.show()