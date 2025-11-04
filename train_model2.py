import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------
# Load processed dataset
# ------------------------
data = pd.read_csv("sleep_apnea_dataset.csv")

# ------------------------
# Add realistic noise to features
# ------------------------
for col in ['heart_rate', 'spo2', 'hr_mean', 'spo2_mean']:
    noise = np.random.normal(0, 1.5, len(data))
    data[col] = data[col] + noise

# Simulate physiological overlaps
apnea_mask = data['apnea_label'] == 1
normal_mask = data['apnea_label'] == 0

data.loc[apnea_mask, 'heart_rate'] -= np.random.uniform(0, 5, apnea_mask.sum())
data.loc[normal_mask, 'heart_rate'] += np.random.uniform(0, 5, normal_mask.sum())

# ------------------------
# Split dataset by patient
# ------------------------
patient_ids = data['patient_id'].unique()
train_ids, test_ids = train_test_split(patient_ids, test_size=0.3, random_state=42)

train_data = data[data['patient_id'].isin(train_ids)].copy()
test_data = data[data['patient_id'].isin(test_ids)].copy()

X_train = train_data.drop(columns=['apnea_label', 'patient_id'])
y_train = train_data['apnea_label'].copy()
X_test = test_data.drop(columns=['apnea_label', 'patient_id'])
y_test = test_data['apnea_label'].copy()

# ------------------------
# Introduce small label flip
# ------------------------
flip_indices = train_data.sample(frac=0.07, random_state=42).index
y_train.loc[flip_indices] = 1 - y_train.loc[flip_indices]

# ------------------------
# Random Forest model
# ------------------------
model = RandomForestClassifier(
    n_estimators=25,
    max_depth=2,
    min_samples_split=8,
    class_weight='balanced', 
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------
# Evaluate
# ------------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\n✅ Test Accuracy: {acc:.3f}\n")
print("Confusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# ------------------------
# 📊 Combined Figure: Confusion Matrix + Heatmap
# ------------------------
fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Left: Text display
ax[0].axis('off')
ax[0].text(0.1, 0.8, "Confusion Matrix (Numbers):", fontsize=12, fontweight='bold')
ax[0].text(0.1, 0.6, str(cm), fontsize=12, family='monospace')
ax[0].text(0.1, 0.3, f"Accuracy: {acc:.3f}", fontsize=12)
ax[0].text(0.1, 0.15, "Classification Report:\n" + report, fontsize=8, family='monospace')

# Right: Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Normal (0)', 'Apnea (1)'],
            yticklabels=['Normal (0)', 'Apnea (1)'], ax=ax[1])
ax[1].set_title("Confusion Matrix Heatmap")
ax[1].set_xlabel("Predicted Label")
ax[1].set_ylabel("True Label")

plt.suptitle("Sleep Apnea Detection Results", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("confusion_matrix_combined.png", dpi=300)
plt.show()

# ------------------------
# Save model
# ------------------------
joblib.dump(model, "sleep_apnea_model.pkl")
print("\n💾 Model saved as sleep_apnea_model.pkl")
print("🖼️ Combined confusion matrix image saved as confusion_matrix_combined.png")
