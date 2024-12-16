# Imports
from PIL import Image
import os
import json

def generate_annotation_file(crops_folder, output_file):
    with open(output_file, 'w') as f:
        for file in os.listdir(crops_folder):
            if file.endswith('.jpg'):
                img_path = os.path.join(crops_folder, file)
                label_path = img_path.replace('.jpg', '.txt')

                # Read the corresponding text label
                with open(label_path, 'r') as label_file:
                    label = label_file.read().strip()

                # Write image path and label to the annotation file
                f.write(f"{img_path}\t{label}\n")

# File paths for annotation files
train_annotation_txt = "PassportDataset/train/train_annotation.txt"
test_annotation_txt = "PassportDataset/test/test_annotation.txt"
# train_annotation_txt = "/content/drive/My Drive/OCRTraining/PassportDataset/train/train_annotation.txt" # For Google Drive
# test_annotation_txt = "/content/drive/My Drive/OCRTraining/PassportDataset/test/test_annotation.txt"    # For Google Drive

# Generate annotation files
generate_annotation_file("PassportDataset/train/crops", train_annotation_txt)
generate_annotation_file("PassportDataset/test/crops", test_annotation_txt)
# generate_annotation_file("/content/drive/My Drive/OCRTraining/PassportDataset/train/crops", train_annotation_txt) # For Google Drive
# generate_annotation_file("/content/drive/My Drive/OCRTraining/PassportDataset/test/crops", test_annotation_txt)   # For Google Drive

print("Annotation files generated!")
