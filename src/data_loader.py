import os
from torch.utils.data import Dataset
from PIL import Image

class MultiModalBreastDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.modalities = [
            ('CXR', 'Chest_XRay_MSI'),
            ('Histo', 'Histopathological_MSI'),
            ('Ultrasound', 'Ultrasound Images_MSI')
        ]
        self.samples = self._gather_samples()

    def _gather_samples(self):
        # Assumes data_dir/CXR, data_dir/Histo, data_dir/Ultrasound each contain patient folders
        patient_ids = set()
        for _, modality_dirname in self.modalities:
            modality_dir = os.path.join(self.data_dir, modality_dirname)
            if os.path.exists(modality_dir):
                patient_ids.update(os.listdir(modality_dir))
        samples = []
        for pid in patient_ids:
            sample = {}
            for mod, modality_dirname in self.modalities:
                img_path = os.path.join(self.data_dir, modality_dirname, pid, 'image.png')
                if os.path.exists(img_path):
                    sample[mod] = img_path
                else:
                    sample[mod] = None
            sample['patient_id'] = pid
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        images = {}
        for mod, _ in self.modalities:
            img_path = sample[mod]
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                images[mod] = img
            else:
                images[mod] = None
        return {'patient_id': sample['patient_id'], 'images': images} 