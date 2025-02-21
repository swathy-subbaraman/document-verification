from flask import Flask, request, jsonify
from twilio.rest import Client
import random
import time

app = Flask(__name__)

# Store OTPs in a dictionary (temporary storage)
otp_store = {}

# Twilio Credentials (Replace with your actual credentials)
ACCOUNT_SID = "AC6484be3d67a26abd1c525a2c5455d630"
AUTH_TOKEN = "0adc5f2238247eb6d578b036ec06ada7"
TWILIO_PHONE_NUMBER = "+17753682693"

# Initialize Twilio Client
client = Client(ACCOUNT_SID, AUTH_TOKEN)

# OTP Expiry Time (in seconds)
OTP_EXPIRY = 300  # 5 minutes



@app.route("/send_otp", methods=["POST"])
def send_otp():
    data = request.json
    phone_number = data.get("phone_number")

    if not phone_number:
        return jsonify({"error": "Phone number is required"}), 400

    # Generate 6-digit OTP
    otp = str(random.randint(100000, 999999))

    # Store OTP with timestamp
    otp_store[phone_number] = {"otp": otp, "timestamp": time.time()}

    # Send OTP via Twilio SMS
    try:
        message = client.messages.create(
            body=f"Your OTP is: {otp}. It is valid for 5 minutes.",
            from_=TWILIO_PHONE_NUMBER,
            to=phone_number
        )
        return jsonify({"message": "OTP sent successfully", "otp": otp})  # Remove OTP in production
    except Exception as e:
        return jsonify({"error": f"Failed to send OTP: {str(e)}"}), 500



@app.route("/verify_otp", methods=["POST"])
def verify_otp():
    data = request.json
    phone_number = data.get("phone_number")
    entered_otp = data.get("otp")

    if not phone_number or not entered_otp:
        return jsonify({"error": "Phone number and OTP are required"}), 400

    # Check if OTP exists
    if phone_number not in otp_store:
        return jsonify({"error": "Invalid or expired OTP"}), 400

    stored_otp_data = otp_store[phone_number]
    stored_otp = stored_otp_data["otp"]
    timestamp = stored_otp_data["timestamp"]

    # Check OTP expiration
    if time.time() - timestamp > OTP_EXPIRY:
        del otp_store[phone_number]
        return jsonify({"error": "OTP expired"}), 400

    # Check if OTP matches
    if stored_otp == entered_otp:
        del otp_store[phone_number]  # Remove OTP after successful verification
        return jsonify({"message": "OTP verified successfully"}), 200
    else:
        return jsonify({"error": "Invalid OTP"}), 400


if __name__ == "__main__":
    app.run(debug=True)
