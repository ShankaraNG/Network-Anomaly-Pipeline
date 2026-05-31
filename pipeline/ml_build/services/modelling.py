import torch
import torch.nn as nn
import torch.optim as optim


class NetworkAnomalyNN(nn.Module):

    def __init__(self):

        super(NetworkAnomalyNN, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(4, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 12),
            nn.BatchNorm1d(12),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(12, 8),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):

        return self.model(x)