import torch.nn as nn

class AutoROI(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x