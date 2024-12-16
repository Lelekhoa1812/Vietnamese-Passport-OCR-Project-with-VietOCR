from vietocr.tool.predictor import Predictor
from PIL import Image
import os

# Load the trained model and configuration
predictor = Predictor(config)

# Path to the test crops folder
test_crops_folder = '/content/drive/My Drive/OCRTraining/PassportDataset/test/crops'
test_images = [f for f in os.listdir(test_crops_folder) if f.endswith('.jpg')]

correct = 0
total = len(test_images)

for img_name in test_images:
    img_path = os.path.join(test_crops_folder, img_name)
    txt_path = img_path.replace('.jpg', '.txt')

    with open(txt_path, 'r') as f:
        ground_truth = f.read().strip()

    # Load the image using PIL
    image = Image.open(img_path)
    prediction = predictor.predict(image).strip()

    print(f"GT: {ground_truth}, Predicted: {prediction}")
    if prediction == ground_truth:
        correct += 1

# Accuracy per cent
accuracy = correct / total * 100
print(f"Test Accuracy: {accuracy:.2f}%")
