import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_rate=0.5):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Conv
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_size, kernel_size=(3, input_size), stride=1, padding=(1, 0))
        self.bn1 = nn.BatchNorm2d(hidden_size)
        self.conv2 = nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 1), stride=1, padding=(1, 0))
        self.bn2 = nn.BatchNorm2d(hidden_size)
        
        # LSTM
        self.lstm = nn.LSTM(hidden_size + input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
        # FC
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        x = x.unsqueeze(1)

        residual = x.squeeze(1)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x.squeeze(3)
        x = x.permute(0, 2, 1)
        x = torch.cat((x, residual), dim=2)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.bn3(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn4(out)
        out = torch.relu(out)
        out = self.fc3(out)
        return out
