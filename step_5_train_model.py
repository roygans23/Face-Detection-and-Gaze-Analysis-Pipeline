import torch
from config.config_step_5 import (
    ACTORS_READY_FOLDER,
    FACES_FOLDER,
    MODEL_OUTPUT_PATH,
    BATCH_SIZE,
    NUM_CLASSES,
    LEARNING_RATE,
    NUM_EPOCHS,
    DEVICE,
    SIMILARITY_THRESHOLD
)
from modules.model_training import (
    create_dataloaders,
    create_model,
    train_model
)
from modules.feature_extraction import (
    generate_actor_embeddings,
    match_and_move_faces
)

def main():
    print("Step 5: Training the face recognition model")
    
    # 0. Preprocess: Match and move faces
    print("Preprocessing: Matching and moving faces...")
    actor_embeddings = generate_actor_embeddings(ACTORS_READY_FOLDER)
    match_and_move_faces(
        faces_folder=FACES_FOLDER,
        actors_ready_folder=ACTORS_READY_FOLDER,
        actor_embeddings=actor_embeddings,
        threshold=SIMILARITY_THRESHOLD
    )
    
    # 1. Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        data_dir=ACTORS_READY_FOLDER,
        batch_size=BATCH_SIZE
    )
    
    # 2. Initialize model, criterion, and optimizer
    print("Initializing model...")
    model = create_model(
        num_classes=NUM_CLASSES,
        device=DEVICE
    )
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
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
        num_epochs=NUM_EPOCHS
    )
    
    # 4. Save the trained model
    print(f"Saving model to {MODEL_OUTPUT_PATH}")
    torch.save(
        trained_model.state_dict(),
        MODEL_OUTPUT_PATH
    )
    print("Training completed successfully!")

if __name__ == "__main__":
    main() 