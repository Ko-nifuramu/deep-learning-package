import numpy as np
import torch
import torch.nn as nn

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)

def output_padding(h_in, conv_out, padding, kernel_size, stride):
    return h_in - (conv_out - 1) * stride + 2 * padding - (kernel_size - 1) - 1

def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def output_padding_shape(h_in, conv_out, padding, kernel_size, stride):
    return tuple(
        output_padding(h_in[i], conv_out[i], padding, kernel_size, stride)
        for i in range(len(h_in))
    )


class VAE(nn.Module):
    def __init__(self, z_dim, device, _activation=nn.ReLU()):
        super(Vision_VAE, self).__init__()
        self.device = device
        self.encoder = VAE_Encoder(z_dim)
        self.decoder = VAE_Decoder(z_dim)
        self.z_dim = z_dim

        self.var = 1e-2

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        y = self.decoder(z)
        return z, y

    def reparameterize(self, mean, var):
        z = mean + torch.mul(
            torch.sqrt(torch.exp(var)),
            torch.normal(mean=0, std=1, size=mean.shape).to(self.device),
        )
        return z

    def cal_loss(self, x, criterion=nn.MSELoss()):
        scale_adjust = x.shape[1] * x.shape[2] * x.shape[3] / self.z_dim
        mean, log_var = self.encoder.forward(x)
        
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder.forward(z)
        x = x.view(x.size(0), -1)
        x_hat = x_hat.view(y.size(0), -1)

        # 変分下限Lの最大化　-> -Lの最小化
        reconstruction = criterion(y, x) * scale_adjust / (self.var * 2)
        kl = (
            -self.beta * torch.mean(1 + log_var - mean**2 - torch.exp(log_var)) / 2
        )  # beta_vae
        # print("reconstruction : {}".format(reconstruction.shape))
        # print("KL : {}".format(kl.shape))
        loss = reconstruction + kl
        return loss


class VAE_Encoder(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        z_dim,
        activation=nn.ReLU(),
    ):
        super(VAE_Encoder, self).__init__()
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            _activation,
        )  # ->(8, 32, 32)

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            _activation,
        )  # ->(32, 16, 16)

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.BatchNorm2d(num_features=64),
            _activation,
        )  # ->(32, 7, 7)

        self.downsample = nn.Sequential(
            nn.Linear(64 * 7 * 7, 2 * z_dim),
            nn.BatchNorm1d(num_features=2 * z_dim),
            _activation,
        )
        self.mean_layer = nn.Sequential(nn.Linear(2 * z_dim, z_dim))
        self.log_var_layer = nn.Sequential(
            nn.Linear(2 * z_dim, z_dim),
        )

    def forward(self, x):
        out = self.cnn_layer1(x)
        # out = self.SENet(out)
        out = self.cnn_layer2(out)
        out = self.cnn_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.downsample(out)

        mean = self.mean_layer(out)
        log_var = self.log_var_layer(out)

        return mean, log_var


class VAE_Decoder(nn.Module):
    def __init__(self, z_dim):
        super(VAE_Decoder, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(z_dim, 2 * z_dim),
            _activation,
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2 * z_dim, 64 * 4 * 4),
            _activation,
        )

        self.cnn_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=32),
            _activation,
        )  # ->(32, 8, 8)

        self.cnn_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=16),
            _activation,
        )  # ->(16, 16, 16)

        self.cnn_layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16, out_channels=3, kernel_size=4, stride=2, padding=1
            ),
        )  # ->(3, 32, 32)

    def forward(self, z):
        out = self.fc1(z)
        out = self.fc2(out)

        out = out.view(-1, 64, 4, 4)
        out = self.cnn_layer1(out)
        out = self.cnn_layer2(out)
        out = self.cnn_layer3(out)
        out = torch.sigmoid(out)

        return out
