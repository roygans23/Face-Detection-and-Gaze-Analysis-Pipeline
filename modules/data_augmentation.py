# data_augmentation.py

import os
from PIL import Image
from torchvision import transforms

def augment_actors_images(actors_dir, output_base_dir, num_images_threshold=200, num_augmented=10):
    """
    Augments images for actors with fewer than `num_images_threshold` images.
    """
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor()
    ])

    for actor in os.listdir(actors_dir):
        actor_dir = os.path.join(actors_dir, actor)
        if not os.path.isdir(actor_dir):
            continue

        images = [img for img in os.listdir(actor_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= num_images_threshold:
            print(f"Skipping {actor}: {len(images)} images already exist.")
            continue

        print(f"Augmenting images for {actor} (current count: {len(images)})...")
        output_actor_dir = os.path.join(output_base_dir, f"{actor}_augmented")
        os.makedirs(output_actor_dir, exist_ok=True)

        for img_name in images:
            img_path = os.path.join(actor_dir, img_name)
            with Image.open(img_path) as img:
                # Convert RGBA -> RGB if needed
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                for i in range(num_augmented):
                    augmented_tensor = transform(img)
                    # Convert back to PIL
                    augmented_img = transforms.ToPILImage()(augmented_tensor)
                    augmented_img.save(os.path.join(
                        output_actor_dir,
                        f"{os.path.splitext(img_name)[0]}_aug_{i}.jpg"
                    ))
    print("Augmentation completed.")
