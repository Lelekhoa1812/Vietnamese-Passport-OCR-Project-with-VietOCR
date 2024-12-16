# CUSTOM AUGMENTOR
from imgaug import augmenters as iaa
from vietocr.loader.aug import ImgAugTransform
import cv2
import numpy as np
from PIL import Image

# New Augmentor version allow unfixed orientation whether input id image could be rotated by any non-arranged angles, i.e., 90, 180 and 270 deg
class MyAugmentor(ImgAugTransform):
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.Affine(rotate=(-10, 10)),  # Small random rotations
            iaa.GaussianBlur(sigma=(0, 0.1)),  # Blur intensity reduced
            iaa.AdditiveGaussianNoise(scale=(0, 0.01 * 255)),  # Noise intensity lowered
        ])
    
    # Detect if image is rotated in any of the 90, 180 or 270 degrees as the dataset may not be in fixed orientation
    def correct_orientation_cv2(self, img):
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
            if avg_angle > 90:
                avg_angle -= 180
            return avg_angle
        return 0

    def __call__(self, img):
        angle = self.correct_orientation_cv2(img)
        img = img.rotate(-angle, expand=True) if abs(angle) > 1 else img

        # Apply augmentations and return as PIL Image
        augmented = self.aug(image=np.array(img))  # Convert PIL to NumPy, apply augmentation
        return Image.fromarray(augmented)  # Convert back to PIL Image  


# TRAIN DATASET
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

# Can choose vgg_transformer or vgg_seq2seq to load the config
config = Cfg.load_config_from_name('vgg_transformer')
'''
Summary of chosen config model:

| Feature                         | **vgg_transformer**                      | **vgg_seq2seq**                   |
|---------------------------------|------------------------------------------|-----------------------------------|
| **Decoder Type**                | Transformer                              | Seq2Seq with LSTM/GRU             |
| **Speed**                       | Slower on short text, faster on long text| Faster on short text              |
| **Accuracy**                    | Higher for longer text and complex data  | Slightly lower for long sequences |
| **Resource Usage**              | High (Memory, GPU)                       | Lower (Memory, GPU)               |
| **Best for**                    | Long text, complex sequences             | Short text, limited hardware      |
'''

# Can change default character and symbol, however, it has been satisfied and covered the majorities already
# Character and symbol not configured will be set as error
# config['vocab'] = 'tập vocab'

# SAMPLE
# dataset_params = {
#     'name':'hw', # Dataset name
#     'data_root':'./data_line/', # Folder containing annotation and images
#     'train_annotation':'train_line_annotation.txt', # train folder with img and annotation
#     'valid_annotation':'test_line_annotation.txt' # test folder with img and annotation
# }
# PROJECT USAGE
dataset_params = {
    'name': 'passport',  # Dataset name
    'data_root': 'PassportDataset',  # Root folder
    # 'data_root': '/content/drive/My Drive/OCRTraining/PassportDataset',  # For Google Drive
    'train_annotation': 'train/train_annotation.txt',  # Train annotation file
    'valid_annotation': 'test/test_annotation.txt'     # Validation annotation file
}

# SAMPLE
# params = {
#          'print_every':200, # Print loss each 200 iteration
#          'valid_every':10000, # Validate accuracy each 10000 iteraction
#           'iters':20000, # Train iterate 20000 times
#           'export':'./weights/transformerocr.pth', # Save pretrained model
#           'metrics': 10000 # Use 10000 imgs from the test folder to values / set metrics for the model
#          }
# PROJECT USAGE
params = {
    'print_every': 100,  # Print loss every 100 iterations
    'valid_every': 500,  # Validate every 500 iterations
    'iters': 5000,  # Number of iterations for training
    'export': './weights/passportocr.pth',  # Path to save the model
    # 'export': '/content/drive/My Drive/OCRTraining/PassportDataset/weights/passportocr.pth',  # For Google Drive
    'metrics': 20  # Number of test samples used for evaluation
}

# Update custom config as preferation
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0' # Device to train model, change to cpu if cannot use cuda

# Train the model from pretrained model will produce better outcome, especially casing small dataset
# To use custom augmentation, can use Trainer(config, pretrained=True, augmentor=MyAugmentor()) with the given example.
trainer = Trainer(config, pretrained=True, augmentor=MyAugmentor())

trainer.visualize_dataset()  # Optional: visualize dataset samples with augmentations
trainer.train()  # Train the model
trainer.visualize_prediction()  # Optional: visualize model predictions after training
# trainer.config.save('/content/drive/My Drive/OCRTraining/PassportDataset/config.yml')  # For Google Drive
trainer.config.save('config.yml')  # Save configuration for later use
