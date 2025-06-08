import json, torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class GTSRBDataset(Dataset):
    """Load (img path, label) from JSON"""
    def __init__(self, json_path, project_root, train=True):
        self.entries = json.load(open(json_path))
        self.root = Path(project_root)
        # Create transforms without lambda functions for multiprocessing compatibility
        transforms = [T.Resize((64, 64)), T.ToTensor(), T.Normalize([0.5]*3, [0.5]*3)]
        if train:
            transforms = [T.RandomHorizontalFlip(), T.RandomRotation(10)] + transforms
        self.transform = T.Compose(transforms)
    def __len__(self): return len(self.entries)
    def __getitem__(self, idx):
        rec = self.entries[idx]
        img = Image.open(self.root / rec["img"]).convert("RGB")
        return self.transform(img), rec["label"]
