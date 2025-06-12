from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_data_transforms(model_type='vgg'):
    if model_type == 'resnet':
        image_size = (160, 160)
        mean = [0.5] * 3
        std = [0.5] * 3
    elif model_type == 'vgg':
        image_size = (224, 224)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    return {
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

def create_dataloaders(data_dir, batch_size=32, model_arch='vgg'):
    data_transforms = get_data_transforms(model_arch)
    dataset = datasets.ImageFolder(root=data_dir, transform=data_transforms['train'])
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    train_dataset.dataset.transform = data_transforms['train']
    val_dataset.dataset.transform = data_transforms['val']

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    )
