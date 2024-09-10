import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from ..registry import EEGDiffDR  # Assuming this is part of your project structure

@EEGDiffDR.register_module()
class evaluationDataset(Dataset):
    def __init__(self, csv_path, window_size=16, step_size=1):   ## 没有用
        """
        Args:
            csv_path (str): Path to the CSV file containing the dataset.
            window_size (int): The size of the window for each data item.
            step_size (int): The step size to move the window for the next data item.
        """
        self.transform = None
        self.csv_path = csv_path
        self.window_size = window_size
        self.step_size = step_size
        data = pd.read_csv(csv_path, skip_blank_lines=True)
        self.data = data.values[:, 1:window_size+1]  # Assuming you're interested in these columns
        self.normalized_data, self.max_value, self.min_value = self.normalize_with_min_max(self.data)
        # Adjust length calculation for the sliding window
        self.length = max(0, (self.normalized_data.shape[0] - window_size) // step_size + 1)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Adjust index to account for step size
        start_index = index * self.step_size
        #print(start_index) ##enable this for debugging
        image = self.normalized_data[start_index:start_index + self.window_size, :]
        image = torch.from_numpy(image).unsqueeze(0).float()
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
