import torch
import torch.nn as nn


class conditional_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(z_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, img_size * img_size), nn.Tanh()
        )

    def forward(self, x, c):
        batch_size = x.shape[0]
        c = c.unsqueeze(1).expand(x.size())
        x = torch.cat((x, c), dim=1)
        return self.G(x).view(batch_size, 1, self.img_size, self.img_size)


class conditional_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Linear(img_size * img_size * 2, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x, c):
        batch_size = x.shape[0]
        out = x.view(batch_size, -1)
        c = c.unsqueeze(1).expand(out.size())
        return self.D(torch.cat((out, c), dim=1))
