from torch.utils.data import Dataset
from PIL import Image
import os
import torch


def parse_data(lines):
    pairs = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 3:
            # Positive pair: (person, img1, img2)
            pairs.append((parts[0], parts[0], int(parts[1]), int(parts[2]), 1))
        elif len(parts) == 4:
            # Negative pair: (person1, img1, person2, img2)
            pairs.append((parts[0], parts[2], int(parts[1]), int(parts[3]), 0))
    return pairs



class FacePairsDataset(Dataset):
    def __init__(self, pairs, root_dir, transform=None):
        """
        pairs: List of tuples (person1, person2, img1, img2, label)
        root_dir: Directory with all the images.
        transform: Optional transform to be applied on a sample.
        """
        self.pairs = pairs
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        person1, person2, img1, img2, label = self.pairs[idx]
        path1 = os.path.join(self.root_dir, person1, f"{person1}_{img1:04d}.jpg")
        path2 = os.path.join(self.root_dir, person2, f"{person2}_{img2:04d}.jpg")
        try:
            image1 = Image.open(path1).convert('L')  # Convert image to grayscale
            image2 = Image.open(path2).convert('L')  # Convert image to grayscale
        except IOError:
            print(f"Error opening one of the images at index {idx}: {path1} or {path2}")
            return None 
        
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)

        return image1, image2, torch.tensor(label, dtype=torch.float32)
