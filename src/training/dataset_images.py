# src/training/dataset_images.py
from pathlib import Path
import cv2
import numpy as np
from torch.utils.data import Dataset

def label_from_filename(fname: str) -> str:
    # expected filename like "gihan_face0.png" or "person1_face2.png"
    base = Path(fname).stem
    # take prefix before "_face" (handles both "_face" and "_face0", "_face1", etc.)
    if "_face" in base:
        # Extract just the first part (person name) before any other underscores
        full_prefix = base.split("_face")[0]
        return full_prefix.split("_")[0]
    # fallback: everything before first underscore
    return base.split("_")[0]

class FaceImageDataset(Dataset):
    """
    Loads aligned PNG face previews (data/processed/*.png).
    Applies Albumentations transforms (train only).
    Returns: augmented_image (numpy HWC, uint8), label (str), original_path (str)
    """
    def __init__(self, images_dir: str, transform=None):
        self.images_dir = Path(images_dir)
        self.files = sorted([p for p in self.images_dir.glob("*.png") if "_face" in p.name])
        self.transform = transform

        # build label map
        labels = [label_from_filename(p.name) for p in self.files]
        unique = sorted(set(labels))
        self.label2idx = {l:i for i,l in enumerate(unique)}
        self.idx2label = {i:l for l,i in self.label2idx.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = cv2.imread(str(p))  # BGR uint8
        if img is None:
            raise RuntimeError(f"Failed to read {p}")
        # Albumentations expects HWC BGR
        if self.transform:
            aug = self.transform(image=img)
            img = aug['image']
        label_str = label_from_filename(p.name)
        label = self.label2idx[label_str]
        return img, label, str(p)
