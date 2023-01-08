import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(128, 64, bias=False)
        self.fc2 = nn.Linear(64, 32, bias=False)
        self.fc3 = nn.Linear(32, 16, bias=True)

        self.fc4 = nn.Linear(16, 32, bias=True)
        self.fc5 = nn.Linear(32, 64, bias=False)
        self.fc6 = nn.Linear(64, 128, bias=False)

    def encoder(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    def decoder(self, x):
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)

        return x

    def forward(self, x):

        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, z
