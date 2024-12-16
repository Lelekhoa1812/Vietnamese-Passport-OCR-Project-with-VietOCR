# Prepare the Dataset for VietOCR
'''
# VietOCR requires training data in a specific format:
Each image should have its corresponding ground truth text in a .txt file.
The annotation format must align with the structure expected by the OCR model.
# Will need to do this
Extract Text Annotations: Parse the JSON annotations to extract the bbox (bounding boxes) for each text field, crop those regions from the images, and associate them with their respective text labels (e.g., firstname, lastname, dob).
'''

# Imports
from PIL import Image
import os
import json

# File paths
dataset_path = "PassportDataset"
train_folder = os.path.join(dataset_path, "train")
test_folder = os.path.join(dataset_path, "test")
train_folder_img = os.path.join(train_folder, "img")
test_folder_img = os.path.join(test_folder, "img")
train_annotation_path = os.path.join(train_folder, "annotation.json")
test_annotation_path = os.path.join(test_folder, "annotation.json")

# Ensure folders exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)
os.makedirs(train_folder_img, exist_ok=True)
os.makedirs(test_folder_img, exist_ok=True)

# Load train and test annotations
with open(train_annotation_path, 'r') as f:
    train_data = json.load(f)
with open(test_annotation_path, 'r') as f:
    test_data = json.load(f)

# Function to crop and save annotations
def crop_and_save_annotations(images_folder, annotations, output_folder, json_data):
    os.makedirs(output_folder, exist_ok=True)
    for anno in annotations:
        img_id = anno["image_id"]
        category_id = anno["category_id"]
        bbox = anno["bbox"]  # [x, y, width, height]

        # Match image file name and category name
        image_file = next(img["file_name"] for img in json_data["images"] if img["id"] == img_id)
        img_path = os.path.join(images_folder, image_file)
        label_name = next(cat["name"] for cat in json_data["categories"] if cat["id"] == category_id)

        if os.path.exists(img_path):
            img = Image.open(img_path)
            # Crop the image based on bounding box
            cropped = img.crop((bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]))
            cropped_img_path = os.path.join(output_folder, f"{img_id}_{label_name}.jpg")
            cropped.save(cropped_img_path)

            # Save corresponding text file
            with open(cropped_img_path.replace(".jpg", ".txt"), "w") as f:
                f.write(label_name)

# Crop and save train and test annotations
crop_and_save_annotations(train_folder_img, train_data["annotations"], os.path.join(train_folder, "crops"), train_data)
crop_and_save_annotations(test_folder_img, test_data["annotations"], os.path.join(test_folder, "crops"), test_data)

print("Data preparation complete!")