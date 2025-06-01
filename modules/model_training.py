# model_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import numpy as np
from modules.fine_tuned_inception_resnet import FineTuneInceptionResnet
from facenet_pytorch import InceptionResnetV1
from utils.tensorboard_utils import TensorboardLogger

def create_dataloaders(data_dir, batch_size=32, model_arch='vgg'):
    """
    Creates train and validation DataLoaders from the dataset folder using ImageFolder structure.
    """
    print(f"Creating DataLoaders for model architecture: {model_arch}...")
    data_transforms = get_data_transforms(model_arch)

    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_data_transforms(model_type='vgg'):
    if model_type == 'resnet':
        image_size = (160, 160)
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        print("Using ResNet data transforms with image size 160x160, mean [0.5, 0.5, 0.5], std [0.5, 0.5, 0.5]")
    elif model_type == 'vgg':
        image_size = (224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        print("Using VGG data transforms with image size 224x224, mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]")
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]),
    }
    return data_transforms

def create_model(num_classes = 1000, device='cpu', model_arch='vgg', model_checkpoint_path=None, pretrained=True, train_mode=True, freeze_grads=False, data_parallel_patch=False):
    
    """
    Loads a pre-trained VGG16 and modifies the final layer to match num_classes.
    """
    print(f"Creating model with architecture: {model_arch}, num_classes: {num_classes}, device: {device}, pretrained: {pretrained}, Mode: {'Train' if train_mode else 'Eval'}, freeze_grads: {freeze_grads}...")
    
    model = build_base_model(model_arch, num_classes, pretrained, train_mode, model_checkpoint_path)

    # Load checkpoint if provided (on base model)
    if model_checkpoint_path:
        print(f"Loading model checkpoint from '{model_checkpoint_path}'")
        model_checkpoint = torch.load(model_checkpoint_path, map_location=device)

        # Check if model checkpoint was wrapped in DataParallel, if so align the base model accordingly
        if data_parallel_patch:
            print("Applying DataParallel patch to model features...")
            model.features = torch.nn.DataParallel(model.features)

        # Load state_dict of model checkpoint, if included
        checkpoint_state_dict = model_checkpoint['state_dict'] if 'state_dict' in model_checkpoint else model_checkpoint

        if train_mode:
            # Remove classifier head weights before loading (to avoid shape mismatch)
            print("Removing classifier head...")
            checkpoint_state_dict = remove_classifier_head(checkpoint_state_dict, model_arch)

            missing, unexpected = model.load_state_dict(checkpoint_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {missing}")
            if unexpected:
                print(f"Unexpected keys: {unexpected}")
            
            print(f"Classifier head removed. Now replacing it with new head for training with {num_classes} outputs...")
            override_classifier_head(model, num_classes, model_arch)
        else:
            # For inference, load the model state_dict directly
            print("Adjusting classifier head of base model to the same shape as in checkpoint for evaluation...")
            adjust_classifier_for_eval(model, checkpoint_state_dict, model_arch)
            
            try:
                model.load_state_dict(checkpoint_state_dict, strict=True)
            except RuntimeError as e:
                print(f"Error loading state dict strictly:\n{e}")

            print(f"Adjusted classifier head for evaluation using checkpoint's output shape.")

        # Display metadata from the checkpoint
        display_checkpoint_metadata(model_checkpoint)

    # Freeze model's base layer gradients if specified (to keep pre-trained 'learned' features intact), except classifier head
    if freeze_grads and train_mode:
        print("Freezing gradients for all layers except classifier head.")
        freeze_base_layer_grads(model, model_arch)

    # Display parameter gradient calculation status
    display_params_grad_calc_status(model)

    # Move to device
    model = model.to(device)
    return model

def train_model(model, criterion, optimizer, train_loader, val_loader, device, num_epochs=20, lr_scheduler=None):
    """
    Trains the model using the specified DataLoaders.
    """

    print("Starting model training...")

    if lr_scheduler:
        print("Using OneCycleLR scheduler for learning rate adjustment.")

    # Create tensorboard writer
    with TensorboardLogger('tensorboard_logs/citizen4_LabPretrainedVGG16_FINETUNED') as writer:
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

                if lr_scheduler:
                    # Update learning rate
                    lr_scheduler.step()

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

            print(f"Current Learning Rate: {lr_scheduler.get_last_lr()[0] if lr_scheduler else optimizer.param_groups[0]['lr']:.6f}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.4f}")

            # Log model layer weights to check if they're learnable
            log_param_norms(model, writer, epoch)

            # Log to TensorBoard
            writer.log_scalar('train/train_loss (per epoch)', train_loss, epoch+1)
            writer.log_scalar('train/train_accuracy (per epoch)', train_acc.item(), epoch+1)
            writer.log_scalar('val/val_loss (per epoch)', val_loss, epoch+1)
            writer.log_scalar('val/val_accuracy (per epoch)', val_acc.item(), epoch+1)
            print(f"Written Loss+Acc logs to Tensorboard")

    print("Training completed.")
    return model

def build_base_model(arch, num_classes, pretrained, train_mode, model_checkpoint_path):
    if arch == 'vgg':
        model = models.vgg16(pretrained=pretrained)
        print(f"Initialized VGG16 with pretrained={pretrained}, num_classes={num_classes}")
        if train_mode:
            override_classifier_head(model, num_classes, arch)
    elif arch == 'resnet':
        print(f"Creating standalone InceptionResnetV1 (FaceNet) with classification head: {train_mode}...")
        pretrained_resnet = 'vggface2' if pretrained else False
        model = InceptionResnetV1(pretrained=pretrained_resnet, classify=train_mode, num_classes=num_classes)

        # if not train_mode and model_checkpoint_path is None:
            # print("Creating standalone InceptionResnetV1 (FaceNet) for evaluation without checkpoint (with no custom classification head required)...")

        # pretrained_resnet = 'vggface2' if pretrained else False
        # model = FineTuneInceptionResnet(num_classes=num_classes, pretrained=pretrained_resnet)
        # print(f"Initialized FineTuneInceptionResnet with pretrained={pretrained_resnet}, num_classes={num_classes}")
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")
    return model

def override_classifier_head(model, num_classes, arch):
    if arch == 'vgg':
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    elif arch == 'resnet':
        in_features = model.logits.in_features
        model.logits = nn.Linear(in_features, num_classes)

def remove_classifier_head(state_dict, arch):
    if arch == 'vgg':
        return {
            k: v for k, v in state_dict.items()
            if not k.startswith('classifier.6')  # works for VGG only
        }
    elif arch == 'resnet':
        return {k: v for k, v in state_dict.items() if not k.startswith('logits')}

def adjust_classifier_for_eval(model, state_dict, arch):
    if arch == 'vgg' and 'classifier.6.weight' in state_dict:
        out_features = state_dict['classifier.6.weight'].shape[0] # Retrieves number of output features from model checkpoint
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, out_features)

    elif arch == 'resnet' and 'logits.weight' in state_dict:
        out_features = state_dict['logits.weight'].shape[0]
        print(f"Adjusting classifier head for evaluation with {out_features} output features.")
        model.logits = nn.Linear(model.logits.in_features, out_features)

def display_checkpoint_metadata(model_checkpoint):
    if 'acc' in model_checkpoint:
        print(f"Model accuracy from checkpoint: {model_checkpoint['acc']:.4f}")

    if 'epoch' in model_checkpoint:
        print(f"Model epoch from checkpoint: {model_checkpoint['epoch']}")

    if 'optimizer' in model_checkpoint:
        print(f"Optimizer state from checkpoint: {model_checkpoint['optimizer']['param_groups']}")

def freeze_base_layer_grads(model, arch):
    """
    Freezes all model parameters.
    """
    # Optionally freeze gradients of all layers except classifier
    print("Freezing gradients for all layers except classifier head.")

    if arch == 'vgg':
        classifier_head_name = 'classifier.6'
    elif arch == 'resnet':
        classifier_head_name = 'logits'
    else:
        raise ValueError(f"Unsupported model architecture: {arch}")

    for name, param in model.named_parameters():
        if classifier_head_name not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

def display_params_grad_calc_status(model):
    """
    Displays the gradient calculation status of each parameter in the model.
    """
    print("Parameter gradient calculation status:")
    for name, param in model.named_parameters():
        print(f"Param {name} has gradient calculation {'ENABLED' if param.requires_grad else 'DISABLED'}")

def log_param_norms(model, writer, epoch):
    for name, param in model.named_parameters():
        module_name = name.split('.')[0]  # Get the top-level module name, e.g., 'logits'
        param_norm = param.data.norm(2).item()
        writer.log_scalar(f'param_norms/{module_name}/{name}', param_norm, epoch + 1)