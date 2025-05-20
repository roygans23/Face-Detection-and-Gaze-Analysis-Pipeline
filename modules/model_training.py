# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np


def create_dataloaders(data_dir, batch_size=32):
    """
    Creates train and validation DataLoaders from the dataset folder using ImageFolder structure.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]),
    }

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def create_model(num_classes=48, device='cpu', model_checkpoint_path=None, pretrained=True):
    """
    Loads a pre-trained VGG16 and modifies the final layer to match num_classes.
    """

    # Load a pre-trained VGG16 model
    model = models.vgg16(pretrained=pretrained)

    # Load checkpoint if provided
    if model_checkpoint_path:
        print(f"Loading model checkpoint from {model_checkpoint_path}")
        model_checkpoint = torch.load(model_checkpoint_path, map_location=device)
        print(f"Checkpoint keys: {model_checkpoint.keys() if isinstance(model_checkpoint, dict) else 'not a dict'}")

        # Align the classifier layer to the number of classes of the custom model path (classifier.6)
        if 'state_dict' in model_checkpoint and 'classifier.6.weight' in model_checkpoint['state_dict']:
            num_output_classes = model_checkpoint['state_dict']['classifier.6.weight'].shape[0]
        elif 'classifier.6.weight' in model_checkpoint:
            num_output_classes = model_checkpoint['classifier.6.weight'].shape[0]
        else:
            # If the classifier layer is not found, use the default num_classes
            print("No classifier layer found in checkpoint, using default num_classes.")
            num_output_classes = num_classes

        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_output_classes)
        print(f"Output Classifier layer modified to {num_output_classes} output classes.")

        # Handle different checkpoint formats
        if isinstance(model_checkpoint, dict):
            if 'state_dict' in model_checkpoint:
                model.load_state_dict(model_checkpoint['state_dict'], strict=False)
            else:
                model.load_state_dict(model_checkpoint, strict=False)
                print("Loaded model checkpoint without 'state_dict' key.")
        else:
            model.load_state_dict(model_checkpoint, strict=False)
            print("Loaded model checkpoint directly.")

        if 'acc' in model_checkpoint:
            print(f"Model accuracy from checkpoint: {model_checkpoint['acc']:.4f}")

        if 'epoch' in model_checkpoint:
            print(f"Model epoch from checkpoint: {model_checkpoint['epoch']}")

        if 'optimizer' in model_checkpoint:
            print(f"Optimizer state from checkpoint: {model_checkpoint['optimizer']['param_groups']}")

    # Move to device
    model = model.to(device)
    return model


def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=20):
    """
    Trains the model using the specified DataLoaders.
    """
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training phase
        model.train()
        train_loss = 0.0
        train_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)

        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_corrects.double() / len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

    print("Training completed.")
    return model
