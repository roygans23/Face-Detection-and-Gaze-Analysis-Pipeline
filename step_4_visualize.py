import os
import matplotlib.pyplot as plt
import pandas
from modules.visualize import load_and_process_data, create_gaze_visualization
from config.config_step_4 import *

def main():
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process all CSV files in the CSV_OUTPUTS directory
    for csv_file in os.listdir(CSV_OUTPUTS):
        if csv_file.endswith('_with_gazes.csv'):
            print(f"Processing {csv_file}...")
            
            # Construct full paths
            csv_path = os.path.join(CSV_OUTPUTS, csv_file)
            output_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(csv_file)[0]}_visualization.png")
            
            try:
                # Load and process the data
                df = load_and_process_data(csv_path, FRAME_SKIP)
                
                # Create visualization
                plt = create_gaze_visualization(df, ACTORS_READY_FOLDER, FRAME_SKIP)
                
                # Save the plot
                plt.savefig(output_path, bbox_inches='tight', dpi=300)
                plt.close()
                
                print(f"Saved visualization to {output_path}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    main()
