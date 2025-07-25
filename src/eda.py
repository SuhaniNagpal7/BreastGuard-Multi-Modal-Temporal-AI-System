import sys
import os
import matplotlib.pyplot as plt
from data_loader import MultiModalBreastDataset

def main():
    data_dir = '../MultiModel Breast Cancer MSI Dataset'
    if not os.path.exists(data_dir):
        data_dir = './MultiModel Breast Cancer MSI Dataset'
    dataset = MultiModalBreastDataset(data_dir)
    print(f'Total patients: {len(dataset)}')
    modality_counts = {m: 0 for m in ['CXR', 'Histo', 'Ultrasound']}
    for sample in dataset.samples:
        for m in modality_counts:
            if sample[m]:
                modality_counts[m] += 1
    print('Modality counts:', modality_counts)

    # Show a few samples
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for j, m in enumerate(['CXR', 'Histo', 'Ultrasound']):
            img = sample['images'][m]
            if img is not None:
                axs[j].imshow(img)
                axs[j].set_title(m)
            else:
                axs[j].set_title(f'{m} (missing)')
            axs[j].axis('off')
        plt.suptitle(f'Patient: {sample["patient_id"]}')
        plt.show()

if __name__ == '__main__':
    main() 