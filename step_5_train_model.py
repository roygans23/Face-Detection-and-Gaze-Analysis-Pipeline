import torch
from torch.optim.lr_scheduler import OneCycleLR
from config.config_step_5 import *

from config.config_step_2 import FACE_RECOGNITION_MODEL_CONFIG

from config.model_config import ModelConfig
from config.inference_model_config import InferenceModelConfig
from config.trainer_config import TrainerConfig

from modules.model_builder import ModelBuilder
from utils.data_utils import create_dataloaders
from modules.model_training import train_model

from modules.feature_extraction import FeatureExtraction

import os

def main():
    print("Step 5: Training the face recognition model")
    feature_extraction = FeatureExtraction(model_config=FACE_RECOGNITION_MODEL_CONFIG)
    
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

    model_builder = ModelBuilder(TRAIN_MODEL_CONFIG)
    model = model_builder.initialize_model()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Optionally, add a learning rate scheduler
    lr_scheduler=OneCycleLR(
            optimizer,
            max_lr=LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=NUM_EPOCHS,
            anneal_strategy=ANNEAL_STRATEGY
        )

    trainer_config = TrainerConfig(
    device=DEVICE,
    num_epochs=NUM_EPOCHS,
    criterion=criterion,
    optimizer=optimizer,
    lr_scheduler=lr_scheduler,
    log_dir='tensorboard_logs/refactored_resnet_pretrained')

    # 3. Train the model
    print("Starting training...")

    trained_model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=trainer_config
    )
    
    # 4. Save the trained model
    print(f"Saving model to {MODEL_OUTPUT_PATH}")

    # Ensure the directory exists
    os.makedirs(MODELS_DIR, exist_ok=True)

    torch.save(trained_model.state_dict(), MODEL_OUTPUT_PATH)
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 