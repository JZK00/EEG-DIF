import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from ..registry import EEGDiffDR

@EEGDiffDR.register_module()
class EEGDataset(Dataset):
    def __init__(self, csv_path):
        self.transform = None
        self.csv_path = csv_path
        data = pd.read_csv(csv_path, skip_blank_lines=True)
        self.data = data.values[:, 1:17]  ## input from excel, 30 or 56 or 16
        self.normalized_data, \
        self.max_value, \
        self.min_value = self.normalize_with_min_max(self.data)

    def __len__(self):
        return self.data.shape[0] - 16 + 1  ## here

    def __getitem__(self, index):
        image = self.normalized_data[index:index + 16, :] ## here
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = image.float()
        if self.transform:
            image = self.transform(image)
        return (image,)

    def normalize_with_min_max(self, data):
        max_values = np.max(data, axis=0)
        min_values = np.min(data, axis=0)
        equal_columns = np.where(max_values == min_values)[0]
        normalized_data = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            if i in equal_columns:
                normalized_data[:, i] = 0.0
            else:
                normalized_data[:, i] = (data[:, i] - min_values[i]) / (max_values[i] - min_values[i])
        return normalized_data, max_values, min_values

    def denormalize_with_min_max(self, normalized_data, max_values, min_values):
        denormalized_data = normalized_data * (max_values - min_values) + min_values
        return denormalized_data


@EEGDiffDR.register_module()
class Long_predictionEEGDataset(Dataset):
    def __init__(self, csv_path):
        self.transform = None
        self.csv_path = csv_path
        data = pd.read_csv(csv_path, skip_blank_lines=True)
        self.data = data.values[:, 1:]
        self.normalized_data, \
        self.max_value, \
        self.min_value = self.normalize_with_min_max(self.data)

    def __len__(self):
        return self.data.shape[0] - 640 + 1

    def __getitem__(self, index):
        image = self.normalized_data[index:index + 640, :]
        image = torch.from_numpy(image)
        image = torch.unsqueeze(image, 0)
        image = image.float()
        if self.transform:
            image = self.transform(image)
        return (image,)

    def normalize_with_min_max(self, data):
        max_values = np.max(data, axis=0)
        min_values = np.min(data, axis=0)
        equal_columns = np.where(max_values == min_values)[0]
        normalized_data = np.zeros_like(data, dtype=float)
        for i in range(data.shape[1]):
            if i in equal_columns:
                normalized_data[:, i] = 0.0
            else:
                normalized_data[:, i] = (data[:, i] - min_values[i]) / (max_values[i] - min_values[i])
        return normalized_data, max_values, min_values

    def denormalize_with_min_max(self, normalized_data, max_values, min_values):
        denormalized_data = normalized_data * (max_values - min_values) + min_values
        return denormalized_data