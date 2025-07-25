
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from peft import get_peft_model, LoraConfig, TaskType
from PIL import Image

# Settings
DATA_DIR = "./MultiModel Breast Cancer MSI Dataset/Chest_XRay_MSI"
MODEL_NAME = "google/vit-base-patch16-224-in21k"
BATCH_SIZE = 4
EPOCHS = 1
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./models"
os.makedirs(SAVE_DIR, exist_ok=True)

# Preprocessing
processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Dataset
class CXRFolderDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform
        for label, class_name in enumerate(['Normal', 'Malignant']):
            class_dir = os.path.join(data_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(class_dir, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

# DataLoader
dataset = CXRFolderDataset(DATA_DIR, transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=2)
model.to(DEVICE)

# Training loop (minimal)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
model.train()
for epoch in range(EPOCHS):
    for imgs, labels in dataloader:
        imgs, labels = imgs.to(DEVICE), torch.tensor(labels).to(DEVICE)
        outputs = model(pixel_values=imgs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Save model
model.save_pretrained(SAVE_DIR)
print("Model saved.")
