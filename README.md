# Passport OCR Scanner with VietOCR Transformer

## Project Overview
This project implements an OCR (Optical Character Recognition) scanner for extracting key fields from passport images, such as names, date of birth, country, gender, and passport number. The system leverages the power of [VietOCR](https://github.com/pbcquoc/vietocr) for training and testing on a custom dataset formatted in COCO-style annotations.

The project includes:
- **Data Preparation**: Splitting the dataset into training and testing sets, generating synthetic passport data, cropping text regions from images, correcting their orientation, and creating annotation files compatible with VietOCR.
- **OCR Model Training**: Training the OCR model on synthetic passport fields using the `vgg_transformer` configuration.
- **Synthetic Data Generation**: A hybrid approach using Python scripts to generate **5000 synthetic passport samples**.
- **Input Orientation Handling**: Automatic correction of image orientation (90°, 180°, 270° rotations) to ensure proper alignment for OCR.
- **Evaluation**: Evaluating model performance on the test dataset to measure accuracy.

## Installation

### Step 1: Install Dependencies
Install the required Python packages using:
```bash
pip install -r requirements.txt
```

Or using pip3:
```bash
pip3 install -r requirements.txt
```

### Step 2: Generate Synthetic Passport Data
To create a dataset of **5000 synthetic passports**:

#### 1. **Synthetic Passport Generation** (2000 Generic Passports):
Use `SyntheticPassportGeneration.py` to generate generic passports with random names, dates of birth, countries, and other fields:
```bash
python SyntheticPassportGeneration.py
```
- Fields include: fullname, nationality, date of birth, place of birth, gender, and passport ID.
- 2000 samples are generated and annotated with bounding boxes for OCR.

#### 2. **Vietnamese Passport Generation** (1500 Printed and 1500 Handwritten):
Use `VietnamesePassportGeneration.py` to generate **3000 Vietnamese passports** with both printed and handwritten text:
```bash
python VietnamesePassportGeneration.py
```
- Handwritten samples use handwriting-like fonts for added variability.
- Fields are generated using random Vietnamese text to simulate realistic passport content.

#### 3. **Add Real-World Imperfections**:
To simulate real-world inconsistencies, such as rotation, noise, and lighting effects, use `GenerateImperfectionMutant.py`:
```bash
python GenerateImperfectionMutant.py
```
This script applies the following transformations to synthetic images:
- **Blurred**: Simulates out-worn or scanned passports.
- **Angled**: Rotates the image to a random diagonal angle.
- **Brightened**: Simulates over or under-lighting conditions.
- **Mixed**: Applies a combination of 2 or 3 techniques mentioned, blur, angle, and brightness adjustments.

The final dataset consists of:
- **2000 Generic Passports**
- **1500 Printed Vietnamese Passports**
- **1500 Handwritten Vietnamese Passports**

Total: **5000 synthetic passport samples**

### Step 3: Data Preparation
Prepare the dataset for OCR training:
- Split the dataset into training and testing sets (80% training, 20% testing).
- Generate cropped text regions and corresponding ground-truth annotations.

Run the provided scripts to organize and preprocess the dataset.

### Step 4: Preprocess and Correct Image Orientation
Use the following script to correct image orientations:
```python
python preprocess_images.py
```

### Step 5: Train the VietOCR Model
Train the OCR model using the prepared dataset:
```python
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

config = Cfg.load_config_from_name('vgg_transformer')

# Update dataset paths
config['dataset'].update({
    'name': 'passport',
    'data_root': '/path/to/PassportDataset',
    'train_annotation': 'train/train_annotation.txt',
    'valid_annotation': 'test/test_annotation.txt'
})

# Training parameters
config['trainer'].update({
    'print_every': 100,
    'valid_every': 500,
    'iters': 5000,
    'export': './weights/passportocr.pth',
    'metrics': 20
})

config['device'] = 'cuda:0'
trainer = Trainer(config, pretrained=True)
trainer.train()
```

### Step 6: Evaluate the Model
Evaluate the trained model using the test dataset and provided evaluation scripts. Results include accuracy metrics and visualized predictions.

## Inference with Orientation Correction
Use the following script to perform inference with real-time orientation correction:
```python
python inference_orientation_correct.py
```

## Project Structure
```plaintext
├──PassportDataset/
│   ├── train/
│       ├── crops/
│       └── train_annotation.txt
│   ├── test/
│       ├── crops/
│       └── test_annotation.txt
├── synthetic_passports/   # 5000 generated synthetic passports
│   └── labels.txt                   # All labels of the faked data generated previously.
│   └── passport_index.jpg           # Generic passport
│   └── vpassport_index.jpg          # Vietnamese passport (both printed and handwritten)
│   └── template           
│         └── passport_template.jpg  # Template background passport img blurred selective fields
│         └── passport_template.json # Labelme annotation for the passport img with polygonized regions
│         └── VietnamesePassportGeneration.py
│         └── SyntheticPassportGeneration.py
│         └── GenerateImperfectionMutant.py
├── mutated_passports/     # Imperfection mutants (noise, lighting, etc.)
├── weights/
│   └── passportocr.pth    # Trained VietOCR model weights
├── patrick_hand_font/     # Hand written font
├── requirements.txt       # Necessary imports
├── README
└── LICENSE
```

## Analysis of Training Results
1. **Training and Validation Loss**:
   - Training loss stabilizes after initial iterations.
   - Validation loss remains consistent, indicating good generalization.

2. **Accuracy**:
   - **Full Sequence Accuracy**: Final accuracy ~97%.
   - **Per-Character Accuracy**: Final accuracy ~97%.

3. **Synthetic Data Impact**:
   - The hybrid dataset (synthetic + augmented imperfections) significantly improved accuracy.

## Key Dependencies
- **VietOCR**: Deep learning-based OCR framework.
- **Pillow**: Image processing for generating synthetic passports.
- **Faker**: Random data generation (names, dates, etc.).
- **OpenCV**: Image rotation and transformation.
- **imgaug**: Augmentation for imperfection mutants.
- **PyTorch**: Training deep learning models.


## Google Collab Notebook
[Training Collab URL](https://colab.research.google.com/drive/1sZmpSJiAb6y3ciqwRzJPdgVEjk7bLZt3?usp=sharing)

[Synthetic Passport Generation](https://colab.research.google.com/drive/1Al4w8ccJxCnMTSFeYEuMXewpzhBO8AgF?usp=sharing)

## Google Drive Dataset
[Dataset](https://drive.google.com/drive/folders/1WUuXciJYsFgsY81KnibXRKDxninKg0bu?usp=sharing)