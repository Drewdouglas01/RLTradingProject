import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self, number_filter, number_of_stocks, seq_len, number_feature, drop):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, number_filter, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(number_filter, number_filter, kernel_size=(3, number_of_stocks))
        self.pool1 = nn.MaxPool2d((1, 2))
        self.conv3 = nn.Conv2d(number_filter, number_filter, kernel_size=(3, 1))
        self.pool2 = nn.MaxPool2d((1, 2))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(drop)
        self.fc = nn.Linear(number_filter * (number_of_stocks * seq_len) // 4, 1)  # Adjust the size accordingly, should be 104 x 1 for 5x60x82

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = torch.sigmoid(self.fc(x))
        return x