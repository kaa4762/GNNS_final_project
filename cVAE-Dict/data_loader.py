import pickle
import torch
from torch.utils.data import Dataset

class PadChestDataset(Dataset):
    def __init__(self, path="../data/dataset_small.pkl", transform=None):
        with open(path, "rb") as f:
            raw_data = pickle.load(f)

        self.data = []
        self.transform = transform

        for sample in raw_data:
            filename, image, projection, diagnosis = sample

            if "PA" in projection:
                proj_idx = 0
            elif "AP" in projection:
                proj_idx = 1
            else:
                continue  # skip unknown projections like 'L' etc.

            self.data.append((filename, image, proj_idx, diagnosis))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, image, proj_idx, diagnosis = self.data[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0
        cond = torch.tensor(proj_idx, dtype=torch.long)

        return {
            "image": image,
            "cond": cond,
            "target": diagnosis,
            "filename": filename
        }
