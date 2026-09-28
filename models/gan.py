import torch
import torch.nn as nn


class vanilla_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, img_size * img_size), nn.Tanh()
        )

    def forward(self, x):
        out = self.G(x)
        return out.view(x.shape[0], 1, self.img_size, self.img_size)


class vanilla_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.D = nn.Sequential(
            nn.Linear(img_size * img_size, 1024), nn.LeakyReLU(0.2),
            nn.Linear(1024, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 128), nn.LeakyReLU(0.2),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.D(x.view(x.shape[0], -1))


class G_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.criterion = nn.BCELoss()

    def forward(self, fake):
        return self.criterion(fake, torch.ones_like(fake).to(self.device))


class D_Loss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.criterion = nn.BCELoss()

    def forward(self, D_real, D_fake):
        d_real_loss = self.criterion(D_real, torch.ones_like(D_real).to(self.device))
        d_fake_loss = self.criterion(D_fake, torch.zeros_like(D_fake).to(self.device))
        return d_real_loss + d_fake_loss, d_real_loss, d_fake_loss
