from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg

config = Cfg.load_config_from_file('config.yml') # Use pre-trained config epxported previously if exist
# config = Cfg.load_config_from_name('vgg_transformer') # Default config
config['weights'] = './weights/transformerocr.pth' # Path to pre-trained weight from previous step
config['device'] = 'cuda:0' # Device runnong on 'cuda:0', 'cuda:1', 'cpu'

detector = Predictor(config)

img = './a.JPG'
img = Image.open(img)
# Prediction step
s = detector.predict(img, return_prob=False) # To convert to prediction probabiltiy, change to "return_prob=True"