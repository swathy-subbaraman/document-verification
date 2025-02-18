from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import torch
import easyocr
import cv2
import re
import numpy as np
from PIL import Image
import requests
import fitz
from torchvision import models, transforms

app = Flask(__name__)
CORS(app)

# Ensure uploads directory exists
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Aadhaar Verification Model
class AadhaarClassifier(torch.nn.Module):
    def __init__(self):
        super(AadhaarClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)  # Binary classification

    def forward(self, x):
        return self.model(x)

model = AadhaarClassifier()
model.load_state_dict(torch.load("aadhaar_classifier1.pth", map_location=torch.device("cpu")))
model.eval()

# OCR Reader
reader = easyocr.Reader(['en'])

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# Function to Check Aadhaar Number
def is_valid_aadhaar(text):
    for word in text:
        clean_word = word.replace(" ", "")
        if clean_word.isdigit() and len(clean_word) == 12:
            return True
    return False

# Function to Check DOB Format
def has_dob(text):
    dob_pattern = r"\bDOB[:\s]*\d{2}/\d{2}/\d{4}\b|\b\d{4}\b"
    return re.search(dob_pattern, ' '.join(text)) is not None

# Function to Detect Human Face
def detect_human_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

# Aadhaar Verification API
@app.route('/verify_aadhaar', methods=['POST'])
def verify_aadhaar():
    image = request.files['image']
    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output, 1)

    text = reader.readtext(image_path, detail=0)

    valid_aadhaar = is_valid_aadhaar(text)
    dob_present = has_dob(text)
    human_face_detected = detect_human_face(image_path)

    if predicted.item() == 1:
        return jsonify({"result": "FAKE Aadhaar Card"})
    elif valid_aadhaar and dob_present and human_face_detected:
        return jsonify({"result": "REAL Aadhaar Card"})
    else:
        return jsonify({"result": "FAKE Aadhaar Card"})

# Extract Images from PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)
        for img in img_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])
    return images

# Scan QR Code
def scan_qr(image_data):
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    value, pts, qr_code = detector.detectAndDecode(img)
    return value if value else None

# Verify URL
def verify_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200 and ".gov.in" in url:
            return "Genuine Death Certificate"
        return "Fake Death Certificate: Invalid or Non-Gov URL"
    except requests.exceptions.RequestException:
        return "Error: Invalid URL"

# Death Certificate Verification API
@app.route('/verify_death', methods=['POST'])
def verify_death():
    pdf = request.files['pdf']
    pdf_path = os.path.join(UPLOAD_FOLDER, pdf.filename)
    pdf.save(pdf_path)

    images = extract_images_from_pdf(pdf_path)
    results = [verify_url(scan_qr(img)) for img in images if scan_qr(img)]

    return jsonify({"results": results if results else ["No QR code detected"]})

if __name__ == '__main__':
    app.run(debug=True)
