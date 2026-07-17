import os
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import kagglehub

ImageFile.LOAD_TRUNCATED_IMAGES = True

def download_cotton_dataset():
    """
    Downloads the SAR-CLD-2024 dataset from Kaggle automatically.
    Returns the path where the dataset is stored.
    """
    path = kagglehub.dataset_download("sabuktagin/dataset-for-cotton-leaf-disease-detection")
    print("Dataset downloaded to path:", path)
    return path

class RobustLeafDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform

        if not os.path.isdir(root_dir):
            raise ValueError(f"Directory {root_dir} does not exist")

        while True:
            items = os.listdir(root_dir)
            dirs = [d for d in items if os.path.isdir(os.path.join(root_dir, d))]
            files = [f for f in items if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
            
            if len(dirs) == 1 and len(files) == 0:
                root_dir = os.path.join(root_dir, dirs[0])
            else:
                break
                
        self.root_dir = root_dir
        self.classes = sorted([d for d in os.listdir(self.root_dir)
                               if os.path.isdir(os.path.join(self.root_dir, d))])

        if not self.classes:
            raise ValueError(f"No valid class directories found in {self.root_dir}")

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = self._build_samples()

        if not self.samples:
            raise ValueError(f"No valid images found in {self.root_dir}")

    def _build_samples(self):
        samples = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.webp', '.bmp', '.jfif')
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            
            for root, _, files in os.walk(cls_dir):
                for img_name in files:
                    if img_name.lower().endswith(valid_extensions):
                        img_path = os.path.join(root, img_name)
                        try:
                            with Image.open(img_path) as img:
                                img.verify()
                            samples.append((img_path, self.class_to_idx[cls]))
                        except Exception as e:
                            pass
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            img_path, label = self.samples[idx]
            with Image.open(img_path) as img:
                img = img.convert('RGB')
                if self.transform:
                    img = self.transform(img)
                    if img is None:
                        raise ValueError("Transform returned None")
                return img, label
        except Exception as e:
            return torch.zeros(3, 224, 224), 0
