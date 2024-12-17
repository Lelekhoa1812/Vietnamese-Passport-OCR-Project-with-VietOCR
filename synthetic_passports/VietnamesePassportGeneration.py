# Using pre-set label and standard passport image background, blurred box to be changed
# annotated with {[x1,y1], [x2,y2], [x3,y3], [x4,y4]} positions
# then generate synthetic data using faker into those selective fields

# STEP 1: Assert locations that need to be changed with according data output
from PIL import Image, ImageDraw, ImageFont
import random
import os
import cv2
import numpy as np
import json
from faker import Faker
from multiprocessing import Pool
import requests
import zipfile

# Initialize Faker and paths
fake = Faker()
root_path = "/content/drive/My Drive/OCRTraining/PassportDataset"
output_folder = os.path.join(root_path, "synthetic_passports")
label_file = os.path.join(output_folder, "labels.txt")
template_image_path = os.path.join(output_folder, "template/passport_template.png")
template_json_path = os.path.join(output_folder, "template/passport_template.json")
font_path = os.path.join(root_path, "DejaVuSans-Bold.ttf")
font_zip_url = "https://github.com/dejavu-fonts/dejavu-fonts/releases/download/version_2_37/dejavu-fonts-ttf-2.37.zip"

os.makedirs(output_folder, exist_ok=True)

# Function to download and extract font
def download_font():
    if not os.path.exists(font_path):
        print("Font not found. Downloading...")
        response = requests.get(font_zip_url, stream=True)
        if response.status_code == 200:
            with open("dejavu-fonts.zip", 'wb') as f:
                f.write(response.content)
            with zipfile.ZipFile("dejavu-fonts.zip", 'r') as zip_ref:
                zip_ref.extractall(root_path)
            os.rename(
                os.path.join(root_path, "dejavu-fonts-ttf-2.37/ttf/DejaVuSans-Bold.ttf"),
                font_path
            )
        else:
            raise ConnectionError("Failed to download font file.")


# STEP 2: Define logics for synthetic data
# nameid section is formatted as VMNYOUR<FULL<Name<
def generate_nameid(fullname):
    return "VNM" + "<".join(fullname.upper().split()) + "<"

# Vietnamese vocabularies listing
vietnamese_vocab = "aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ"

# Randomly picking Vietnamese name from the set of vocabs, without specifically need to set a meaning
def generate_vietnamese_text(length=10):
    """
    Generate random Vietnamese text using the VietOCR vocab.
    """
    return "".join(random.choice(vietnamese_vocab) for _ in range(length))

# Define faking rule
def generate_fields_vietnamese(is_handwritten=False):
    """
    Generate Vietnamese fields for passports.
    Handwritten data is generated using handwriting-like fonts.
    """
    fullname = generate_vietnamese_text(12)   # E.g. Name can be 12 characters
    nationality = generate_vietnamese_text(8) # E.g. Country can be 8 characters
    pob = generate_vietnamese_text(15)        # E.g. POB can be 15 characters
    dob = fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%d/%m/%Y")
    sex = random.choice(["Nam/M", "Nữ/F"])
    cmnd = str(random.randint(100000000, 999999999))
    passportid = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000, 9999999)}"
    nameid = generate_nameid(fullname)

    return {
        "fullname": fullname,
        "nationality": nationality,
        "pob": pob,
        "dob": dob,
        "sex": sex,
        "cmnd": cmnd,
        "passportid": passportid,
        "nameid": nameid,
        "passportid2": passportid
    }

# Convert position coordination from labelme annotation to bbox for text writing
def extract_bbox(points):
    """
    Convert a list of polygon points into a bounding box (x1, y1, x2, y2).
    Assumes points are provided as [[x1, y1], [x2, y2], ...].
    """
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

# Allocate appropriate position for the text to be placed
def place_text(draw, text, bbox, font):
    """
    Place text centered within the bounding box.
    """
    x1, y1, x2, y2 = bbox
    # w, h = draw.textsize(text, font=font)
    # In the latest versions of Pillow (>= 10.0), textbbox is preferred as it provides a bounding box for the text.
    # So we shouldn't use textsize as it could possibly overlapping the bbox configured
    bbox = draw.textbbox((0, 0), text, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # Center the text in within the bbox
    x = x1 + (x2 - x1 - w) / 2  # Center horizontally
    y = y1 + (y2 - y1 - h) / 2  # Center vertically
    draw.text((x, y), text, fill=(0, 0, 0), font=font)

# Init empty list of results for later collection
results = []

# STEP 4: Write the faked data into selective fields, converting json regions to bbox
def generate_passport_vietnamese(index):
    try:
        with open(template_json_path, 'r') as f:
            regions_data = json.load(f)
            regions = {shape["label"]: extract_bbox(shape["points"])
                       for shape in regions_data.get("shapes", [])}

        # Load background image
        background = cv2.imread(template_image_path)
        img = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        # Decide if this passport is handwritten or printed
        is_handwritten = index % 2 == 0  # Alternate between printed and handwritten (e.g., even to be printed and odd to be handwritten)
        font_used = font_path # printed font to be used as Arial
        if is_handwritten:
          font_used = os.path.join(root_path, "patrick_hand_font/PatrickHand-Regular.ttf")
        font = ImageFont.truetype(font_used, size=24)

        # Generate Vietnamese fields
        fields = generate_fields_vietnamese(is_handwritten)

        # Place text into regions
        for label, text in fields.items():
            if label in regions:
                bbox = regions[label]
                place_text(draw, text, bbox, font)

        # Save the generated image
        img_path = os.path.join(output_folder, f"vpassport_{index + 1}.jpg")  # Start from 2001
        img.save(img_path)

        # Save labels
        with open(label_file, "a") as lf:
            lf.write(f"{img_path}|{json.dumps(fields)}\n")

    except Exception as e:
        print(f"Error generating passport {index + 1}: {e}")

# Main
if __name__ == "__main__":
    download_font()
    num_samples = 3000 # The number of faked sample to be generated, change this as the size of volume demanded
    with open(label_file, "w") as lf:
        lf.write("Image_Path|Fields\n")  # Write file header
    # Start multiprocessing
    with Pool(processes=8) as pool:
        pool.map(generate_passport_vietnamese, range(1, num_samples + 1)) # Indexing from i (1) to n+1
    print("Synthetic passport generation complete!")
