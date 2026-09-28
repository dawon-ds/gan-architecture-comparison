import torch.nn as nn


class dcgan_G(nn.Module):
    def __init__(self, z_dim, img_size):
        super().__init__()
        self.img_size = img_size
        self.G = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64, 7, 1, 0, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.G(x.view(x.shape[0], -1, 1, 1))


class dcgan_D(nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.img_size = img_size
        self.D = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        return self.D(x).view(x.shape[0], -1)
