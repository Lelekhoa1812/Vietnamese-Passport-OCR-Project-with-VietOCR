# Using pre-set label and standard passport image background, blurred box to be changed
# annotated with {[x1,y1], [x2,y2], [x3,y3], [x4,y4]} positions
# then generate synthetic data using faker into those selective fields

# STEP 1: Assert locations that need to be changed with according data output
from PIL import Image, ImageDraw, ImageFont
from faker import Faker
import random
import os
import cv2
import numpy as np
import json
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

def generate_fields():
    fullname = fake.name()
    nationality = fake.country()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=65).strftime("%d/%m/%Y")
    sex = random.choice(["Nam/M", "Nữ/F"])
    pob = f"{fake.city()}, {fake.country()}"
    cmnd = str(random.randint(100000000, 999999999))
    passportid = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000, 9999999)}"
    nameid = generate_nameid(fullname)
    # Debug print to ensure fields are generated
    print({
        "fullname": fullname,
        "nationality": nationality,
        "dob": dob,
        "sex": sex,
        "pob": pob,
        "cmnd": cmnd,
        "passportid": passportid,
        "nameid": nameid,
        "passportid2": passportid
    })
    # Return faked data to be saved
    return {
        "fullname": fullname,
        "nationality": nationality,
        "dob": dob,
        "sex": sex,
        "pob": pob,
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
def generate_passport(index):
    try:
        # Load template JSON regions
        with open(template_json_path, 'r') as f:
            regions_data = json.load(f)
            regions = {shape["label"]: extract_bbox(shape["points"])
                       for shape in regions_data.get("shapes", [])}

        # Load background and initialize drawing
        background = cv2.imread(template_image_path)
        img = Image.fromarray(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font_path, size=24)

        # Generate fake data fields
        fields = generate_fields()

        # Draw fields based on JSON regions
        for label, text in fields.items():
            if label in regions:
                bbox = regions[label]  # Retrieve bbox for label
                print(f"Placing '{label}': {text} at {bbox}")
                place_text(draw, text, bbox, font)
            else:
                print(f"Warning: Missing or invalid points for label '{label}'")

        # Save image
        img_path = os.path.join(output_folder, f"passport_{index}.jpg")
        img.save(img_path)

        # Collect path and fields generated from fake data into the result list
        results.append({"path": img_path, "fields": fields})

        # Add noise
        # img_cv = cv2.imread(img_path)
        # noise = np.random.normal(0, 15, img_cv.shape).astype(np.uint8)
        # img_cv = cv2.add(img_cv, noise)
        # cv2.imwrite(img_path, img_cv)
        '''
        The addition of noise in the output image was likely included for simulating real-world imperfections such as:
          - Scanned document artifacts
          - Printing distortions
          - Image sensor noise in photos
        Comment when not needed
        '''
        # Save labels
        with open(label_file, "a") as lf:
            lf.write(f"{img_path}|{json.dumps(fields)}\n")

    except Exception as e:
        print(f"Error generating passport {index}: {e}")

# Main
if __name__ == "__main__":
    download_font()
    num_samples = 5000 # The number of faked sample to be generated, change this as the size of volume demanded
    with open(label_file, "w") as lf:
        lf.write("Image_Path|Fields\n")  # Write file header
    # Start multiprocessing
    with Pool(processes=8) as pool:
        pool.map(generate_passport, range(1, num_samples + 1)) # Indexing from i (1) to n+1
    print("Synthetic passport generation complete!")