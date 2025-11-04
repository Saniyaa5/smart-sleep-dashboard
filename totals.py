# patient_summary_fix.py
import pandas as pd
import numpy as np

# ---- config ----
INPUT_CSV = "sleep_apnea_dataset.csv"   # replace with your generated file
OUT_SUMMARY = "patient_summary.csv"
OUT_APNEA = "apnea_patients.csv"
OUT_NORMAL = "normal_patients.csv"

# Thresholds / rules you can adjust
MEAN_THRESHOLD = 0.2        # patient considered apnea if mean(apnea_label) > 0.2
EVENT_COUNT_THRESHOLD = 10  # patient considered apnea if apnea_event_count >= this
ANY_EVENT = True            # if True, create a label based on any-event rule

# load
df = pd.read_csv(INPUT_CSV)

# sanity
print("Columns:", df.columns.tolist())
print("Total rows:", len(df))
print("Unique patients:", df['patient_id'].nunique())

# per-patient stats
grp = df.groupby('patient_id')['apnea_label']
patient_stats = grp.agg([
    ('apnea_mean','mean'),
    ('apnea_sum','sum'),
    ('n_rows','count')
]).reset_index()

# add candidate patient-level labels
# 1) any-event rule
patient_stats['label_any'] = (patient_stats['apnea_sum'] > 0).astype(int)

# 2) mean-threshold rule
patient_stats['label_mean_thresh'] = (patient_stats['apnea_mean'] > MEAN_THRESHOLD).astype(int)

# 3) event-count rule
patient_stats['label_event_count'] = (patient_stats['apnea_sum'] >= EVENT_COUNT_THRESHOLD).astype(int)

# 4) majority rule (>50%)
patient_stats['label_majority'] = (patient_stats['apnea_mean'] > 0.5).astype(int)

# choose one as final — here I will pick any-event by default
patient_stats['final_label'] = patient_stats['label_any']

# add human-readable condition
patient_stats['condition'] = patient_stats['final_label'].map({0:'Normal Patient', 1:'Apnea Patient'})

# show distribution
print("\nSummary of per-patient apnea fractions:")
print(patient_stats['apnea_mean'].describe())

print("\nCounts by rule:")
print("Any-event:", int(patient_stats['label_any'].sum()), "apnea patients")
print("Mean-thresh (>{}%):".format(int(MEAN_THRESHOLD*100)), int(patient_stats['label_mean_thresh'].sum()))
print("Event-count (>= {}):".format(EVENT_COUNT_THRESHOLD), int(patient_stats['label_event_count'].sum()))
print("Majority (>50%):", int(patient_stats['label_majority'].sum()))

# save summary and separate lists for whichever rule you want
patient_stats.to_csv(OUT_SUMMARY, index=False)
patient_stats[patient_stats['final_label']==1][['patient_id','apnea_mean','apnea_sum','n_rows']].to_csv(OUT_APNEA, index=False)
patient_stats[patient_stats['final_label']==0][['patient_id','apnea_mean','apnea_sum','n_rows']].to_csv(OUT_NORMAL, index=False)

print("\nSaved:", OUT_SUMMARY, OUT_APNEA, OUT_NORMAL)
