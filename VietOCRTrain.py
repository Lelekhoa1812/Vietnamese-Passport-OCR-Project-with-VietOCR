# CUSTOM AUGMENTOR
from imgaug import augmenters as iaa
from vietocr.loader.aug import ImgAugTransform

class MyAugmentor(ImgAugTransform):
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.GaussianBlur(sigma=(0, 1.0)),
            iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
        ])

config['device'] = 'cuda:0'  # Use GPU; change to 'cpu' if GPU is unavailable
trainer = Trainer(config, pretrained=True, augmentor=MyAugmentor())  # Use custom augmentation


# TRAIN DATASET
from vietocr.tool.config import Cfg
from vietocr.model.trainer import Trainer

# Can choose vgg_transformer or vgg_seq2seq to load the config
config = Cfg.load_config_from_name('vgg_transformer')

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
    'train_annotation': 'train/train_annotation.txt',  # Train annotation file
    'valid_annotation': 'test/test_annotation.txt'    # Validation annotation file
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
    'metrics': 20  # Number of test samples used for evaluation
}

# Update custom config as preferation
config['trainer'].update(params)
config['dataset'].update(dataset_params)
config['device'] = 'cuda:0' # Device to train model, change to cpu if cannot use cuda

# Train the model from pretrained model will produce better outcome, especially casing small dataset
# To use custom augmentation, can use Trainer(config, pretrained=True, augmentor=MyAugmentor()) with the given example.
trainer = Trainer(config, pretrained=True)

trainer.visualize_dataset()  # Optional: visualize dataset samples with augmentations
trainer.train()  # Train the model
trainer.visualize_prediction()  # Optional: visualize model predictions after training
trainer.config.save('config.yml')  # Save configuration for later use
