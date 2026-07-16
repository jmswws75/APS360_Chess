from Baseline import ChessDataset, evaluate, train_net
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    A standard Residual Block for an 8x8 grid.
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x # Save the original input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual # SKIP CONNECTION: Add the original input back
        out = F.relu(out)
        return out

class PrimaryModel(nn.Module):
    def __init__(self):
        super(PrimaryModel, self).__init__()
        self.name = "Primary"
        
        self.conv0 = nn.Conv2d(14, 128, 3, padding=1)
        self.bn0 = nn.BatchNorm2d(128)
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )
        
        
        flattened_size = 128 * 8 * 8
        self.fc1 = nn.Linear(flattened_size, 1024)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(1024, 4096)
        
    def forward(self, x):
        x = F.relu(self.bn0(self.conv0(x)))
        x = self.res_blocks(x)
        
        x = x.view(x.size(0), -1) 

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == "__main__":
    
    dataset = ChessDataset("chess_dataset.npz")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    model = PrimaryModel()
    train_net(model, batch_size=64, learning_rate=0.01, num_epochs=20, train_loader=train_loader, val_loader=val_loader)
    