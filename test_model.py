import pandas as pd
import joblib
import os

# -----------------------------
# Config: paths
# -----------------------------
MODEL_PATH = r"C:\Users\SANIYA SULTHANA\FYPROJECT\sleep_apnea_model.pkl"  # trained model
DATASET_CSV = r"C:\Users\SANIYA SULTHANA\FYPROJECT\sleep_apnea_dataset.csv"  # full processed dataset
OUTPUT_FOLDER = r"C:\Users\SANIYA SULTHANA\FYPROJECT\predicted"  # folder to save predictions

# Create folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -----------------------------
# Load model
# -----------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
model = joblib.load(MODEL_PATH)
print("Model loaded successfully.")

# -----------------------------
# Load full dataset
# -----------------------------
if not os.path.exists(DATASET_CSV):
    raise FileNotFoundError(f"Dataset CSV not found: {DATASET_CSV}")
df_all = pd.read_csv(DATASET_CSV)
print(f"Full dataset loaded: {len(df_all)} rows, {df_all['patient_id'].nunique()} patients")

# -----------------------------
# Input multiple patient IDs (comma separated)
# -----------------------------
ids_input = input("Enter patient IDs to test (comma separated): ")
patient_ids = [int(x.strip()) for x in ids_input.split(",")]

# -----------------------------
# Features used for training
# -----------------------------
FEATURES = ['time_sec', 'heart_rate', 'spo2', 'hr_mean', 'spo2_mean', 'hr_std', 'spo2_std', 'hr_diff', 'spo2_diff']

# -----------------------------
# Prepare summary
# -----------------------------
summary_list = []

for pid in patient_ids:
    if pid not in df_all['patient_id'].values:
        print(f"Patient ID {pid} not found. Skipping...")
        continue

    df_patient = df_all[df_all['patient_id'] == pid].copy()
    X_patient = df_patient[FEATURES]
    
    # Predict per row
    df_patient['apnea_pred'] = model.predict(X_patient)
    
    # Summarize per patient
    apnea_fraction = df_patient['apnea_pred'].mean()
    threshold = 0.05  # 5% of night has apnea
    final_label = "Apnea Patient" if apnea_fraction > threshold else "Normal Patient"
    
    summary_list.append({
        "patient_id": pid,
        "n_rows": len(df_patient),
        "predicted_apnea_fraction": round(apnea_fraction, 3),
        "final_label": final_label
    })
    
    # Save row-wise predictions under 'predicted' folder
    output_file = os.path.join(OUTPUT_FOLDER, f"patient_{pid}_prediction.csv")
    df_patient.to_csv(output_file, index=False)
    print(f"Patient {pid}: Row-wise predictions saved to {output_file}")

# -----------------------------
# Save summary for all patients
# -----------------------------
summary_df = pd.DataFrame(summary_list)
summary_file = os.path.join(OUTPUT_FOLDER, "patients_summary_predictions.csv")
summary_df.to_csv(summary_file, index=False)
print("\nSummary of all predictions saved to:", summary_file)
print(summary_df)
