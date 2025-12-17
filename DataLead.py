import os
import shutil
from torchvision import transforms
from PIL import Image
import random

# Base path to original data
base_path = "./data/"  # original data folder

# Augmented data will be saved at the same level as 'data'
augmented_base_path = "./augmented_data/"  # <-- outside ./data/
os.makedirs(augmented_base_path, exist_ok=True)

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
class_counts = {}

# Count original images
for cls in classes:
    folder_path = os.path.join(base_path, cls)
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    class_counts[cls] = len(image_files)
    print(f"{cls}: {len(image_files)} images")

print("\nTotal images before augmentation:", sum(class_counts.values()))

# Find the maximum count to use as target
max_count = max(class_counts.values())
print(f"\nMax class count: {max_count}")
print(f"Target count for all classes: {max_count}\n")

# Define augmentation pipeline
augmentation_transforms = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
])

# Augmentation function
def augment_class_folder(class_name, min_target_count=500):
    src_folder = os.path.join(base_path, class_name)
    dst_folder = os.path.join(augmented_base_path, class_name)  # now inside ./augmented_data/
    os.makedirs(dst_folder, exist_ok=True)

    # List original images
    original_images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    # Copy original images to augmented folder
    for img_name in original_images:
        src_path = os.path.join(src_folder, img_name)
        dst_path = os.path.join(dst_folder, img_name)
        if not os.path.exists(dst_path):  # avoid duplicates
            try:
                # Validate image before copying
                img = Image.open(src_path)
                img.verify()  # Verify it's a valid image
                img.close()
                shutil.copy(src_path, dst_path)
            except Exception as e:
                print(f"Skipped invalid image {src_path}: {e}")

    # Keep augmenting until we reach exactly 500 images
    aug_counter = 1

    # execute at least once
    while True:
        for img_name in original_images:

            if len(os.listdir(dst_folder)) >= min_target_count:
                break

            src_path = os.path.join(src_folder, img_name)
            try:
                img = Image.open(src_path).convert("RGB")
            except Exception as e:
                print(f"Skipped invalid image {src_path}: {e}")
                continue

            augmented_img = augmentation_transforms(img)

            base_name, ext = os.path.splitext(img_name)
            new_name = f"{base_name}_aug{aug_counter}{ext}"
            dst_path = os.path.join(dst_folder, new_name)

            # Skip if file already exists
            if not os.path.exists(dst_path):
                augmented_img.save(dst_path)
            else:
                print(f"Skipped existing file: {dst_path}")

        # Check if reached the target to exit
        if len(os.listdir(dst_folder)) >= min_target_count:
            break

        aug_counter += 1
        print(f"{class_name}: {len(os.listdir(dst_folder))} images (round {aug_counter})")

    print(f"{class_name} augmented to {len(os.listdir(dst_folder))} images.")


# Run for all classes
for cls in classes:
    augment_class_folder(cls, min_target_count=max_count)

print(f"\n✅ All augmented data saved in: {os.path.abspath(augmented_base_path)}")
