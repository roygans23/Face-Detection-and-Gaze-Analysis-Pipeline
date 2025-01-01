# modules/prediction.py

import os
import torch
from PIL import Image
from torchvision import transforms
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def update_dataframe_with_predictions(df, image_folder, dataset_classes, model):
    """
    Updates the DataFrame with a 'name' column based on predictions from the fine-tuned model.

    Parameters:
    - df: The DataFrame containing 'frame', 'index_in_frame', 'top_left', and 'bottom_right' columns.
    - image_folder: Path to the folder containing cropped face images.
    - dataset_classes: List of class names from dataset.classes.
    - model: The fine-tuned PyTorch model (already loaded and on correct device).

    Returns:
    - Updated DataFrame with an additional 'name' column.
    """
    names = []

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to match VGG16 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Ensure the model is in eval mode
    model.eval()

    for _, row in df.iterrows():
        frame = row.name  # Because df.set_index("frame", inplace=True)
        index_in_frame = row['index_in_frame']
        top_left = row['top_left']
        bottom_right = row['bottom_right']

        filename = f"{frame}_{index_in_frame}_{top_left}_{bottom_right}.jpeg"
        filepath = os.path.join(image_folder, filename)

        if os.path.exists(filepath):
            try:
                image = Image.open(filepath).convert("RGB")
                input_tensor = preprocess(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    _, predicted = torch.max(outputs, 1)

                class_name = dataset_classes[predicted.item()]
                names.append(class_name)

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                names.append("Error")
        else:
            names.append("Missing Image")

    df['name'] = names
    return df
