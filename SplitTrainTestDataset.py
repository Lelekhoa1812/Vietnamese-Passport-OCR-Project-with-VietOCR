# Split Train and Test set
# Imports
import json
import random
import os
import shutil

# File paths
dataset_path = "PassportDataset"
images_folder = os.path.join(dataset_path, "img")
annotations_folder = os.path.join(dataset_path, "original_annotation")
train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
train_folder_img = os.path.join(train_folder, "img")
test_folder_img = os.path.join(test_folder, "img")
annotation_path = os.path.join(annotations_folder, "annotation.json")
train_annotation_path = os.path.join(train_folder, "annotation.json")
test_annotation_path = os.path.join(test_folder, "annotation.json")

# Create train and test folders if not exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder_img, exist_ok=True)
os.makedirs(test_folder_img, exist_ok=True)

# Load the annotation.json file
with open(annotation_path, 'r') as f:
    data = json.load(f)

# Get the images list
images = data['images']

# Shuffle the images randomly
random.shuffle(images)

# Split images: 80% train, 20% test
split_index = int(0.8 * len(images))
train_images = images[:split_index]
test_images = images[split_index:]

# Function to filter annotations
def filter_annotations(annotations, image_ids):
    return [anno for anno in annotations if anno['image_id'] in image_ids]

# Get the image IDs for train and test splits
train_image_ids = {img['id'] for img in train_images}
test_image_ids = {img['id'] for img in test_images}

# Filter annotations for train and test
train_annotations = filter_annotations(data['annotations'], train_image_ids)
test_annotations = filter_annotations(data['annotations'], test_image_ids)

# Move images to train and test folders
def move_images(image_list, dest_folder):
    for img in image_list:
        src_path = os.path.join(images_folder, img['file_name'])
        dest_path = os.path.join(dest_folder, img['file_name'])
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

move_images(train_images, train_folder_img)
move_images(test_images, test_folder_img)

# Create train and test JSON structures
train_data = {
    "info": data['info'],
    "licenses": data['licenses'],
    "categories": data['categories'],
    "images": train_images,
    "annotations": train_annotations,
}

test_data = {
    "info": data['info'],
    "licenses": data['licenses'],
    "categories": data['categories'],
    "images": test_images,
    "annotations": test_annotations,
}

# Save train and test JSON files
with open(train_annotation_path, 'w') as f:
    json.dump(train_data, f, indent=4)

with open(test_annotation_path, 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"Dataset split completed:")
print(f"Train dataset: {len(train_images)} images")
print(f"Test dataset: {len(test_images)} images")
