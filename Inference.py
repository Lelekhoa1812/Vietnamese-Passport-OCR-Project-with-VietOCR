from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_file('config.yml') # Use pre-trained config epxported previously if exist
# config = Cfg.load_config_from_file('/content/drive/My Drive/OCRTraining/PassportDataset/config.yml') # For Goolge Drive
# config = Cfg.load_config_from_name('vgg_transformer') # Default config pre-trained from the author of VietOCR
config['weights'] = './weights/transformerocr.pth' # Path to pre-trained weight from previous step
# config['weights'] = '/content/drive/My Drive/OCRTraining/PassportDataset/weights/passportocr.pth' # For Google Drive
config['device'] = 'cuda:0' # Device runnong on 'cuda:0', 'cuda:1', 'cpu'

detector = Predictor(config)

img = './a.JPG'
img = Image.open(img)
# Prediction step
s = detector.predict(img, return_prob=False) # To convert to prediction probabiltiy, change to "return_prob=True"