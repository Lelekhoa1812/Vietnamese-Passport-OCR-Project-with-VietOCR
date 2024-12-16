# Passport OCR Scanner with VietOCR Transformer

## Project Overview
This project implements an OCR (Optical Character Recognition) scanner for extracting key fields from passport images, such as names, date of birth, country, gender, and passport number. The system leverages the power of [VietOCR](https://github.com/pbcquoc/vietocr) for training and testing on a custom dataset formatted in COCO-style annotations.

The project includes:  
- **Data Preparation**: Splitting the dataset into training and testing sets, cropping text regions from images, correcting their orientation, and generating annotation files compatible with VietOCR.
- **OCR Model Training**: Training the OCR model on cropped passport text fields using the `vgg_transformer` configuration.
The project includes:
- **Input Orientation Handling**: Automatic correction of image orientation (90°, 180°, 270° rotations) to ensure proper alignment for OCR.
- **Evaluation**: Evaluating model performance on the test dataset to measure accuracy.


## Installation

### Step 1: Install Dependencies
Install the required Python packages using:
```
pip install -r requirements.txt
```
Or using `pip3`:
```
pip3 install -r requirements.txt
```

### Step 2: Download the Dataset
The original dataset is available at:
[https://github.com/iAmmarTahir/MASK-RCNN-Dataset/tree/master/PakCNIC/Augmented](https://github.com/iAmmarTahir/MASK-RCNN-Dataset/tree/master/PakCNIC/Augmented)

Download and preprocess the dataset following the steps in this repository.

### Step 3: Data Preparation
- Split the dataset into training and testing sets (80% training, 20% testing).
- Extract text annotations based on bounding boxes and save them as cropped images with corresponding `.txt` files for ground truth labels.

Run the provided Python scripts to prepare the dataset:
1. **Dataset Splitting**:
    - Split the dataset into `train` and `test` folders.
    - Save annotations in `train_annotation.txt` and `test_annotation.txt`.

2. **Generate Cropped Text Regions**:
    - Crop text regions from the passport images.
    - Save the cropped images and their labels in the `crops` folder.

### Step 4: Preprocess and Correct Image Orientation
During data preparation, images in the dataset are processed to align their orientation. Use the following script to preprocess all images in the dataset:

```python
import cv2
import os
import numpy as np
from PIL import Image

# Function to detect and correct image orientation
def correct_orientation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Detect dominant text orientation using Hough Line Transform
    def find_orientation(img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        angles = []

        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) % 180
                if 80 <= angle <= 100 or 260 <= angle <= 280:
                    angles.append(angle)
        
        if len(angles) > 0:
            avg_angle = np.mean(angles)
            if avg_angle > 90: avg_angle -= 180
            return avg_angle
        return 0

    # Rotate the image to correct orientation
    angle = find_orientation(img)
    if abs(angle) > 1:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated_img)
    else:
        return Image.open(image_path)

# Apply orientation correction to all images in train and test folders
def preprocess_images(folder):
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img_path = os.path.join(folder, file)
            corrected_img = correct_orientation(img_path)
            corrected_img.save(img_path)

# Paths to images
train_img_folder = '/path/to/PassportDataset/train/img'
test_img_folder = '/path/to/PassportDataset/test/img'

# Correct orientation
preprocess_images(train_img_folder)
preprocess_images(test_img_folder)

print("Image orientation corrected.")
```

### Step 5: Train the VietOCR Model
Train the OCR model using the prepared dataset:
```python
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

dataset_params = {
    'name': 'passport',
    'data_root': '/path/to/PassportDataset',
    'train_annotation': 'train/train_annotation.txt',
    'valid_annotation': 'test/test_annotation.txt'
}

params = {
    'print_every': 100,
    'valid_every': 500,
    'iters': 5000,
    'export': './weights/passportocr.pth',
    'metrics': 20
}

config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0'  # Change to 'cpu' if GPU is not available

trainer = Trainer(config, pretrained=True)
trainer.train()
```

### Step 6: Evaluate the Model
After training, evaluate the model on the test dataset. Use the provided evaluation script to compute accuracy and visualize predictions.


## File Structure (example)
```
PassportDataset/
├── train/
│   ├── crops/
│   │   ├── 1_firstname.jpg
│   │   ├── 1_firstname.txt
│   │   ├── 2_lastname.jpg
│   │   ├── 2_lastname.txt
│   │   ├── ...
│   ├── annotation.json
│   ├── train_annotation.txt
├── test/
│   ├── crops/
│   │   ├── 3_firstname.jpg
│   │   ├── 3_firstname.txt
│   │   ├── 4_lastname.jpg
│   │   ├── 4_lastname.txt
│   │   ├── ...
│   ├── annotation.json
│   ├── test_annotation.txt
```


## Inference with Orientation Correction
Correct image orientation **on-the-fly** during inference using the following script:

```python
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import numpy as np
from PIL import Image

# Load the trained configuration
config = Cfg.load_config_from_file('/path/to/PassportDataset/config.yml')
config['weights'] = '/path/to/PassportDataset/weights/passportocr.pth'
config['device'] = 'cuda:0'

detector = Predictor(config)

# Function to correct image orientation
def correct_orientation(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    def find_orientation(img):
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        edges = cv2.Canny(thresh, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        angles = []

        if lines is not None:
            for rho, theta in lines[:, 0]:
                angle = (theta * 180 / np.pi) % 180
                if 80 <= angle <= 100 or 260 <= angle <= 280:
                    angles.append(angle)

        if len(angles) > 0:
            avg_angle = np.mean(angles)
            if avg_angle > 90: avg_angle -= 180
            return avg_angle
        return 0

    angle = find_orientation(img)
    if abs(angle) > 1:
        (h, w) = img.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        corrected_img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(corrected_img)
    else:
        return Image.open(image_path)

# Correct and predict
image_path = '/path/to/test/image.jpg'
corrected_img = correct_orientation(image_path)
prediction = detector.predict(corrected_img, return_prob=False)
print(f"Predicted Text: {prediction}")
```


## Analysis of Training Results

1. **Training and Validation Loss Trends**:
   - **Training Loss**: Stabilized at around `0.612` after the initial iterations.
   - **Validation Loss**: Fluctuates slightly but remains in the range of `0.600 - 0.650`. This is a positive sign as the validation loss doesn't significantly increase, suggesting no major overfitting.

2. **Accuracy**:
   - **Full Sequence Accuracy (Full Seq)**:
     - Early iterations: `0.8438` (84.38%)
     - Final iteration: `0.9688` (96.88%)
   - **Per-Character Accuracy (Per Char)**:
     - Early iterations: `0.8624` (86.24%)
     - Final iteration: `0.9688` (96.88%)
   - The significant increase in accuracy suggests the model effectively learned the patterns in your dataset.

3. **Validation Results**:
   - The validation accuracy and loss at various points indicate the model generalizes well to unseen data, with consistent improvements in both metrics during training.

4. **Convergence**:
   - Both the training and validation losses have plateaued, indicating the model has likely reached its optimal performance for the current configuration and dataset size.

5. **Learning Rate**:
   - The learning rate decayed steadily during training, starting from `3e-4` to near-zero by the final iteration (`1.24e-9`), which is expected and helps stabilize the model in later training stages.


## Key Dependencies
- **VietOCR**: A deep learning-based OCR framework.
- **Pillow**: Image processing for cropping and saving passport fields.
- **PyTorch**: For training the OCR model.
- **labelme2coco**: To convert dataset annotations into COCO format.
- **imgaug**: Augmentation library for enhancing the dataset.


## Google Collab Notebook
[Collab URL](https://colab.research.google.com/drive/1sZmpSJiAb6y3ciqwRzJPdgVEjk7bLZt3?usp=sharing)


## Acknowledgement of Dataset
The original dataset belongs to:
[iAmmarTahir/MASK-RCNN-Dataset](https://github.com/iAmmarTahir/MASK-RCNN-Dataset/tree/master/PakCNIC/Augmented)
