# STEP 5: Add inconsistent measurements (mutants) simulating real-world imperfections
'''
The set of passport data could be split into different sectional aspects:
- Blurred: Add noise to simulate out-worn physical passport
- Angled: Place the passport in a diagonal direction, not necessarily being in any fixed angle (i.e., right angled)
- Brightening: Under or over lighting or shadowing the image could simulate real-life light situation of the passport can be imperfect
- Mixed: Mixing mutation technique by any 3 of the aboves.
Soft grouping the dataset to process mutation technique applied randomly on selective files
'''

import os
import cv2
import random
import numpy as np
from PIL import Image, ImageEnhance
from multiprocessing import Pool

# Paths
root_path = "/content/drive/My Drive/OCRTraining/PassportDataset"
input_folder = os.path.join(root_path, "synthetic_passports")
output_folder = os.path.join(root_path, "mutated_passports")
os.makedirs(output_folder, exist_ok=True)

# List all input images
input_images = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]

# Mutation techniques
def apply_blur(image):
    """Apply Gaussian blur to simulate out-worn passports."""
    return cv2.GaussianBlur(image, (5, 5), random.uniform(1.0, 3.0))

def apply_angle(image):
    """Rotate image to a random diagonal angle."""
    angle = random.randint(-20, 20)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

def apply_brightness(image):
    """Randomly adjust the brightness of the image."""
    factor = random.uniform(0.5, 1.5)  # Brightness factor between 0.5 (darker) to 1.5 (brighter)
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    enhancer = ImageEnhance.Brightness(pil_image)
    enhanced_image = enhancer.enhance(factor)
    return cv2.cvtColor(np.array(enhanced_image), cv2.COLOR_RGB2BGR)

def apply_mixed(image):
    """Apply any 3 mutations: blurred, angled, brightness."""
    techniques = [apply_blur, apply_angle, apply_brightness]
    random.shuffle(techniques)
    for i in range(3):
        image = techniques[i](image)
    return image

# Mutation dispatcher
def mutate_image(image_path, mutation_type, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    # Apply mutation
    if mutation_type == "blurred":
        mutated_image = apply_blur(image)
    elif mutation_type == "angled":
        mutated_image = apply_angle(image)
    elif mutation_type == "brightened":
        mutated_image = apply_brightness(image)
    elif mutation_type == "mixed":
        mutated_image = apply_mixed(image)
    else:
        print(f"Unknown mutation type: {mutation_type}")
        return
    
    # Save the mutated image
    cv2.imwrite(output_path, mutated_image)

# Worker function for multiprocessing
def process_image(index):
    file_name = input_images[index]
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, file_name)

    # Randomly assign a mutation type
    mutation_type = random.choice(["blurred", "angled", "brightened", "mixed"])
    print(f"Applying '{mutation_type}' to {file_name}")
    mutate_image(input_path, mutation_type, output_path)

# Main function
if __name__ == "__main__":
    num_images = len(input_images)
    print(f"Total images to process: {num_images}")
    
    # Start multiprocessing to process images
    with Pool(processes=8) as pool:
        pool.map(process_image, range(num_images))
    
    print("Mutation process complete! Mutated images are saved.")
