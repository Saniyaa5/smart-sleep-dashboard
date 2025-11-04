import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(r"C:\Users\SANIYA SULTHANA\FYPROJECT\sleep_apnea_dataset.csv")   # change path as needed
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

# -----------------------------
# CHECK FOR MISSING VALUES
# -----------------------------
print("🔍 Missing / Null Values per Column:")
missing = df.isnull().sum()
print(missing)
print(f"\nTotal missing cells: {missing.sum()}")

if missing.sum() > 0:
    print("⚠️ Missing values found! Consider cleaning or imputing them.\n")
else:
    print("✅ No missing values found.\n")

# -----------------------------
# CHECK UNIQUE PATIENTS
# -----------------------------
unique_patients = df['patient_id'].nunique()
print(f"👤 Unique patients in dataset: {unique_patients}")

# -----------------------------
# CHECK PATIENT-WISE LABEL DISTRIBUTION
# -----------------------------
# Assuming 'apnea_label' is 0 = normal, 1 = apnea
patient_labels = df.groupby('patient_id')['apnea_label'].max().reset_index()

# Count apnea vs non-apnea patients
patient_counts = patient_labels['apnea_label'].value_counts()

print("\n⚖️ Patient-Level Label Distribution:")
for label, count in patient_counts.items():
    label_name = "Apnea Patient (1)" if label == 1 else "Normal Patient (0)"
    print(f"  {label_name}: {count} patients")

ratio = patient_counts.min() / patient_counts.max()
print(f"\n📊 Patient-level balance ratio: {ratio:.3f} (closer to 1 = balanced)")

# -----------------------------
# VISUALIZE PATIENT DISTRIBUTION
# -----------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='apnea_label', data=patient_labels, palette='coolwarm')
plt.title('Patient-Level Distribution (Apnea vs Normal)')
plt.xlabel('Apnea Label (0 = Normal, 1 = Apnea)')
plt.ylabel('Number of Patients')
plt.grid(alpha=0.3)
plt.show()

# -----------------------------
# EXTRA CHECK: PER-PATIENT DATA POINT COUNT
# -----------------------------
records_per_patient = df.groupby('patient_id').size()
print("\n📈 Records per patient (min, max, avg):")
print(f"  Min: {records_per_patient.min()} | Max: {records_per_patient.max()} | Avg: {records_per_patient.mean():.2f}")
