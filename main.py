# main.py

import torch
import config
import numpy as np
import pandas as pd

# Import modules
from modules.face_detection import (
    load_caffe_model, 
    process_videos_in_folder
)
from modules.json_to_dataframe import create_dataframes_from_jsons
from modules.face_extraction import process_dataframes
from modules.feature_extraction import (
    generate_actor_embeddings,
    match_and_move_faces
)
from modules.data_augmentation import augment_actors_images
from modules.model_training import (
    create_dataloaders, 
    create_model, 
    train_model
)


def main():
    # 1. Load face detection model
    net = load_caffe_model(config.PROTOTXT_PATH, config.CAFFE_MODEL_PATH)

    # 2. Process videos to JSON
    process_videos_in_folder(
        folder_path=config.VIDEO_FOLDER,
        output_folder=config.JSON_FOLDER,
        net=net,
        frame_skip=config.FRAME_SKIP,
        conf_threshold=config.CONF_THRESHOLD,
        nms_threshold=config.NMS_THRESHOLD
    )

    # 3. Convert JSONs -> DataFrames
    dataframes = create_dataframes_from_jsons(config.JSON_FOLDER)

    # 4. Extract faces from the videos
    process_dataframes(
        dataframes,
        video_folder=config.VIDEO_FOLDER,
        output_base_folder=config.FACES_FOLDER
    )

    # 5. Feature extraction: Build embeddings for known actors
    actor_embeddings = generate_actor_embeddings(config.ACTORS_READY_FOLDER)

    # 6. Match & move faces
    match_and_move_faces(
        faces_folder=config.FACES_FOLDER,
        actors_ready_folder=config.ACTORS_READY_FOLDER,
        actor_embeddings=actor_embeddings,
        threshold=config.THRESHOLD
    )

    # 7. Data augmentation
    augment_actors_images(
        actors_dir=config.ACTORS_READY_FOLDER,
        output_base_dir=config.AUGMENTED_FOLDER,
        num_images_threshold=200,
        num_augmented=3
    )

    # 8. Training process
    train_loader, val_loader = create_dataloaders(
        data_dir=config.DATASET_DIR,
        batch_size=config.BATCH_SIZE
    )

    model = create_model(num_classes=config.NUM_CLASSES, device=config.DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    trained_model = train_model(
        model, 
        criterion, 
        optimizer, 
        train_loader, 
        val_loader, 
        device=config.DEVICE, 
        num_epochs=config.NUM_EPOCHS
    )

    # 9. Save the trained model
    torch.save(trained_model.state_dict(), "fine_tuned_vgg16.pth")
    print("Model saved as fine_tuned_vgg16.pth")

if __name__ == "__main__":
    main()
