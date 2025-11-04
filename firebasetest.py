import pyrebase
import csv

# ===== Firebase Configuration =====
config = {
   "apiKey": "AIzaSyB9DlcQUBcLN9HLpAMg0NPe4MXinSqk-U8",
  "authDomain": "sleep-apnea-98274.firebaseapp.com",
  "databaseURL": "https://sleep-apnea-98274-default-rtdb.firebaseio.com",
  "projectId": "sleep-apnea-98274",
  "storageBucket": "sleep-apnea-98274.firebasestorage.app",
  "messagingSenderId": "509528365321",
  "appId": "1:509528365321:web:9d485df934a171ec661bd8",
  "measurementId": "G-DSP5152QEP"
}


firebase = pyrebase.initialize_app(config)
db = firebase.database()

# ===== Fetch all patient_10 readings =====
patient_id = "patient_10"
data = db.child("SensorData").child(patient_id).get()

# ===== Prepare CSV =====
csv_filename = "patient_10_all_readings.csv"
csv_headers = ["time_sec", "heart_rate", "spo2", "temperature"]

with open(csv_filename, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_headers)

    if data.each():
        # Sort by time_sec key
        for time_key in sorted(data.val().keys(), key=lambda x: int(x)):
            reading = data.val()[time_key]
            hr = reading.get("heart_rate", 0)
            spo2 = reading.get("spo2", 0)
            temp = reading.get("temperature", 0)
            writer.writerow([time_key, hr, spo2, temp])

print(f"All readings exported to {csv_filename} successfully!")