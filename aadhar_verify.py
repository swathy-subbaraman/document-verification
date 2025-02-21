import torch
import cv2
import easyocr
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import re

# Load Trained Model
class AadhaarClassifier(torch.nn.Module):
    def __init__(self):
        super(AadhaarClassifier, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 2)  # Binary classification
    
    def forward(self, x):
        return self.model(x)

# Initialize Model
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

# Function to Check Aadhaar Number (12-digit numeric)
# Function to Check Aadhaar Number (12-digit numeric with spaces)
def is_valid_aadhaar(text):
    aadhaar_number = None
    for word in text:
        # Remove spaces and check if the word consists only of digits and has exactly 12 digits
        clean_word = word.replace(" ", "")
        if clean_word.isdigit() and len(clean_word) == 12:
            aadhaar_number = clean_word
            break
    
    if aadhaar_number:
        print(f"Aadhaar Number Found: {aadhaar_number}")
        return True
    return False


# Function to Check DOB Format (DD/MM/YYYY or YYYY)
def has_dob(text):
    dob_pattern = r"\bDOB[:\s]*\d{2}/\d{2}/\d{4}\b|\b\d{4}\b"
    # Search for the pattern and return True if found
    return re.search(dob_pattern, ' '.join(text)) is not None



# Function to Check if Name is in English
def is_english_name(text):
    return any(word.isalpha() and all(ord(c) < 128 for c in word) for word in text)

# Function to Detect a Human Face
def detect_human_face(image_path):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

# Function to Check for Hidden Content (Basic Tampering Detection)
def has_hidden_content(image_path):
    img = cv2.imread(image_path)
    return cv2.Laplacian(img, cv2.CV_64F).var() < 50  # Low variance may indicate hidden content

# Function to Predict Aadhaar Authenticity
def verify_aadhaar(image_path):
    print(f"\nVerifying Aadhaar for: {image_path}")
    
    # Load Image
    img = Image.open(image_path).convert("RGB")
    img_transformed = transform(img).unsqueeze(0)

    # Model Prediction
    with torch.no_grad():
        output = model(img_transformed)
        _, predicted = torch.max(output, 1)

    result = "REAL Aadhaar Card" if predicted.item() == 0 else "FAKE Aadhaar Card"
    

    # OCR Extraction
    text = reader.readtext(image_path, detail=0)
    

    # Perform Additional Checks
    valid_aadhaar = is_valid_aadhaar(text)
    dob_present = has_dob(text)
    english_name = is_english_name(text)
    human_face_detected = detect_human_face(image_path)
    hidden_content_detected = has_hidden_content(image_path)

    # # Print individual checks
    # if not valid_aadhaar:
    #     print(" Aadhaar number missing or invalid!")
    # if not dob_present:
    #     print(" Date of Birth missing!")
    # if not english_name:
    #     print(" Name is not in English!")
    # if not human_face_detected:
    #     print(" No human face detected!")
    # if hidden_content_detected:
    #     print(" Hidden content detected!")

    # Final Decision
    if predicted.item() == 1:  # Model predicted Fake
        print("Verification Result: FAKE Aadhaar Card")
    else:  # Model predicted Real
        if valid_aadhaar and dob_present and english_name and human_face_detected and not hidden_content_detected:
            print("Verification Result: REAL Aadhaar Card")
        else:
            print("Verification Result: FAKE Aadhaar Card")

# Example Usage
image_path = "C:/Users/swath/Downloads/img.png"  # Change to your test image path
verify_aadhaar(image_path)
