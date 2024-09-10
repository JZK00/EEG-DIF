import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from seizureDataset import SeizureDataset
from model import LSTMClassifier
from tqdm import tqdm
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import os

def collate_fn(batch):
    data, labels = zip(*batch)
    data_padded = pad_sequence(data, batch_first=True, padding_value=0)
    labels = torch.stack(labels, dim=0)

    return data_padded, labels

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_dir = 'path to data'

dataset = SeizureDataset(data_dir)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

model = LSTMClassifier(16, 128, 3, 1, 0.25)
model.load_state_dict(torch.load('path to weight'))
model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

epochs = 150
tolerance = 5  # Number of epochs to tolerate non-decreasing val loss before switching dataloader


model.train()

for epoch in range(epochs):
    
    t_loss = 0
    for data, ground_truth in tqdm(train_loader, leave=False):
        data, ground_truth = data.to(device), ground_truth.to(device)
        optimizer.zero_grad()
        pred = model(data)
        ground_truth = ground_truth.view(-1, 1) 
        ground_truth = ground_truth.float() 
        loss = criterion(pred, ground_truth)
        t_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(t_loss/len(train_loader))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), f'ckpts/weight_lstm_{epoch}.pth')

print("Training complete!")
torch.save(model.state_dict(), 'ckpts/weight_lstm_final.pth')
