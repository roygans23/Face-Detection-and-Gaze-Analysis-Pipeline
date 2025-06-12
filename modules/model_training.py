import torch
from utils.tensorboard_utils import TensorboardLogger
from config.trainer_config import TrainerConfig

def train_model(model, train_loader, val_loader, config: TrainerConfig):
    with TensorboardLogger(config.log_dir) as writer:
        for epoch in range(config.num_epochs):
            print(f"Epoch {epoch+1}/{config.num_epochs}")

            train_loss, train_acc = _run_epoch(
                model, train_loader, config.criterion, config.optimizer, config.device, True, config.lr_scheduler
            )

            val_loss, val_acc = _run_epoch(
                model, val_loader, config.criterion, config.optimizer, config.device, False
            )

            writer.log_scalar('train/loss', train_loss, epoch+1)
            writer.log_scalar('train/accuracy', train_acc.item(), epoch+1)
            writer.log_scalar('val/loss', val_loss, epoch+1)
            writer.log_scalar('val/accuracy', val_acc.item(), epoch+1)

            print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    return model

def _run_epoch(model, loader, criterion, optimizer, device, is_train=True, scheduler=None):
    model.train() if is_train else model.eval()
    epoch_loss, correct = 0.0, 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler:
                    scheduler.step()

        epoch_loss += loss.item() * inputs.size(0)
        correct += torch.sum(preds == labels.data)

    return epoch_loss / len(loader.dataset), correct.double() / len(loader.dataset)