import os
import random
import csv

DATA_ROOT = "./MultiModel Breast Cancer MSI Dataset"
MODALITIES = [
    ("CXR", "Chest_XRay_MSI"),
    ("Histo", "Histopathological_MSI"),
    ("Ultrasound", "Ultrasound Images_MSI")
]
TIMEPOINTS = [0, 6, 12, 18, 24]  # months
OUTPUT_CSV = "./synthetic_temporal_sequences.csv"

# Gather patient IDs from all modalities
patient_ids = set()
for _, modality_dir in MODALITIES:
    modality_path = os.path.join(DATA_ROOT, modality_dir)
    if os.path.exists(modality_path):
        patient_ids.update(os.listdir(modality_path))

# For each patient, randomly assign images to timepoints for each modality
rows = []
for pid in patient_ids:
    for mod, modality_dir in MODALITIES:
        modality_path = os.path.join(DATA_ROOT, modality_dir, pid)
        if not os.path.exists(modality_path):
            continue
        images = [f for f in os.listdir(modality_path) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not images:
            continue
        for t in TIMEPOINTS:
            img = random.choice(images)
            img_path = os.path.join(modality_path, img)
            rows.append({
                "patient_id": pid,
                "modality": mod,
                "timepoint": t,
                "image_path": img_path
            })

# Write to CSV
with open(OUTPUT_CSV, "w", newline="") as csvfile:
    fieldnames = ["patient_id", "modality", "timepoint", "image_path"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)

print(f"Synthetic temporal sequences saved to {OUTPUT_CSV}") 