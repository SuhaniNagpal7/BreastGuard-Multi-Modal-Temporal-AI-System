import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTModel, ViTImageProcessor
from temporal_transformer import TemporalTransformer
from PIL import Image
import numpy as np

DATA_CSV = "./synthetic_temporal_sequences.csv"
BATCH_SIZE = 2
EPOCHS = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 768
NUM_MODALITIES = 3
NUM_TIMEPOINTS = 5
NUM_CLASSES = 3

# Load ViT models for each modality
VIT_MODELS = {
    "CXR": ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE),
    "Histo": ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE),
    "Ultrasound": ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(DEVICE),
}
PROCESSOR = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=PROCESSOR.image_mean, std=PROCESSOR.image_std)
])

class TemporalSequenceDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.patients = self.df['patient_id'].unique()
        self.modality_map = {"CXR":0, "Histo":1, "Ultrasound":2}
        self.time_map = {0:0, 6:1, 12:2, 18:3, 24:4}

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        pid = self.patients[idx]
        patient_df = self.df[self.df['patient_id'] == pid]
        # (time, modalities, embed_dim)
        seq = np.zeros((NUM_TIMEPOINTS, NUM_MODALITIES, EMBED_DIM), dtype=np.float32)
        for _, row in patient_df.iterrows():
            t_idx = self.time_map[row['timepoint']]
            m_idx = self.modality_map[row['modality']]
            img = Image.open(row['image_path']).convert('RGB')
            img_tensor = TRANSFORM(img).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = VIT_MODELS[row['modality']](img_tensor).last_hidden_state[:,0,:].cpu().numpy().squeeze()
            seq[t_idx, m_idx, :] = emb
        # Dummy label: random risk class for each timepoint
        label = np.random.randint(0, NUM_CLASSES, size=(NUM_TIMEPOINTS,))
        return torch.tensor(seq), torch.tensor(label, dtype=torch.long)

dataset = TemporalSequenceDataset(DATA_CSV)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TemporalTransformer(embed_dim=EMBED_DIM, num_modalities=NUM_MODALITIES, num_timepoints=NUM_TIMEPOINTS, num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(EPOCHS):
    for seqs, labels in dataloader:
        seqs, labels = seqs.to(DEVICE), labels.to(DEVICE)
        out = model(seqs)  # (batch, time, num_classes)
        out = out.view(-1, NUM_CLASSES)
        labels = labels.view(-1)
        loss = loss_fn(out, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Temporal transformer training complete.") 