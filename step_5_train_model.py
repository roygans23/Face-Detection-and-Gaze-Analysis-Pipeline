import torch
from torch.optim.lr_scheduler import OneCycleLR
from config.config_step_5 import *
from modules.model_training import (
    create_dataloaders,
    create_model,
    train_model
)

from modules.feature_extraction import FeatureExtraction

import os

def main():
    print("Step 5: Training the face recognition model")

    feature_extraction = FeatureExtraction(
        face_recognition_model_checkpoint=None,
        face_recognition_arch='resnet',
        pretrained=True,
        device=DEVICE,
        embedding_layer_name=RESNET_EMBEDDING_LAYER_NAME
    )
    
    # 0. Preprocess: Match and move faces
    print("Preprocessing: Matching and moving faces...")
    actor_embeddings = feature_extraction.generate_actor_embeddings(ACTORS_READY_FOLDER)
    feature_extraction.match_and_move_faces(
        faces_folder=FACES_FOLDER,
        actors_ready_folder=ACTORS_READY_FOLDER,
        actor_embeddings=actor_embeddings,
        threshold=SIMILARITY_THRESHOLD
    )
    
    # 1. Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=ACTORS_READY_FOLDER,
        batch_size=BATCH_SIZE,
        model_arch='vgg',
    )
    
    # 2. Initialize model, criterion, and optimizer
    print("Initializing model...")
    model = create_model(
        num_classes=NUM_CLASSES,
        device=DEVICE,
        model_arch='vgg',
        model_checkpoint_path=PRETRAINED_VGGFACE_PATH,
        pretrained=True,
        train_mode=True,
        freeze_grads=False,
        data_parallel_patch=True
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # Optionally, add a learning rate scheduler
    lr_scheduler=OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS,
            anneal_strategy=ANNEAL_STRATEGY
        )
    
    # 3. Train the model
    print("Starting training...")
    trained_model = train_model(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        lr_scheduler=lr_scheduler
    )
    
    # 4. Save the trained model
    print(f"Saving model to {MODEL_OUTPUT_PATH}")

    # Ensure the directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    torch.save(trained_model.state_dict(), MODEL_OUTPUT_PATH)
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 