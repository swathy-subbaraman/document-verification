import cv2
import requests
import numpy as np
import fitz  # PyMuPDF for PDF processing

# Function to extract images from a PDF
def extract_images_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        img_list = page.get_images(full=True)
        
        for img in img_list:
            xref = img[0]
            base_image = doc.extract_image(xref)
            img_bytes = base_image["image"]
            images.append(img_bytes)
    
    return images

# Function to scan QR code from an image
def scan_qr(image_data):
    # Convert the image data to an OpenCV image
    np_arr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Initialize QRCode detector
    detector = cv2.QRCodeDetector()

    # Use the detectAndDecode method to detect and decode the QR code
    value, pts, qr_code = detector.detectAndDecode(img)

    if value:
        return value
    else:
        return None

# Function to verify the URL
def verify_url(url):
    try:
        # Send a request to the URL
        response = requests.get(url)
        
        # Check if the URL opens without errors (status code 200)
        if response.status_code == 200:
            # Check if the URL contains ".gov.in"
            if ".gov.in" in url:
                return "Verified: Genuine death certificate"
            else:
                return "Fake death certificate: URL does not belong to a government domain"
        else:
            return "Error: Invalid URL"
    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Main function
def main():
    pdf_path = "C:/Users/swath/Downloads/death.pdf"  # Replace with the path to your PDF
    images = extract_images_from_pdf(pdf_path)
    
    for img_data in images:
        qr_url = scan_qr(img_data)
        if qr_url:
            result = verify_url(qr_url)
            print(result)
        # else:
        #     print("No QR code found in the image")

if __name__ == "__main__":
    main()