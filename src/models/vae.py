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
    def __init__(self, latent_dim, device, _activation=nn.ReLU()):
        super(VAE, self).__init__()
        self.device = device
        self.encoder = VaeEncoder(latent_dim)
        self.decoder = VaeDecoder(latent_dim)
        self.z_dim = latent_dim

        self.var = 1  # 潜在状態の分散

    def forward(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder(z)
        return z, x_hat
    
    def forward_z(self, x):
        mean, log_var = self.encoder(x)
        z = self.reparameterize(mean, log_var)
        return z

    def reparameterize(self, mean, log_var):
        z = mean + torch.mul(
            torch.sqrt(torch.exp(log_var)),
            torch.normal(mean=0, std=1, size=mean.shape).to(self.device),
        )
        return z

    def cal_loss(self, x, criterion=nn.MSELoss()):
        scale_adjust = x.shape[1] * x.shape[2] * x.shape[3] / self.z_dim
        mean, log_var = self.encoder.forward(x)

        z = self.reparameterize(mean, log_var)
        x_hat = self.decoder.forward(z)
        x_hat = x_hat.view(x.size(0), -1)
        x_hat = x_hat.view(x_hat.size(0), -1)

        # 変分下限Lの最大化　-> -Lの最小化
        reconstruction = criterion(x_hat, x) * scale_adjust / (self.var * 2)
        kl = (
            -self.beta * torch.mean(1 + log_var - mean**2 - torch.exp(log_var)) / 2
        )  # beta_vae
        # print("reconstruction : {}".format(reconstruction.shape))
        # print("KL : {}".format(kl.shape))
        loss = reconstruction + kl
        return loss


class VaeEncoder(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        latent_dim: int = 1024,
        activation: nn.Module = nn.Mish,
    ):
        super().__init__()
        self._activation = activation
        self.image_shape = image_shape
        self.latent_dim = latent_dim

        # cnn層の出力サイズ(linear_size)を計算
        conv1_shape = conv_out_shape(
            self.image_shape[1:], paddings[0], kernels[0], strides[0]
        )
        conv2_shape = conv_out_shape(conv1_shape, paddings[1], kernels[1], strides[1])
        conv3_shape = conv_out_shape(conv2_shape, paddings[2], kernels[2], strides[2])
        linear_size = channels[2] * np.prod(conv3_shape).item()

        # 最適化対象のモデル
        self.conv_layer1 = nn.Sequential(
            nn.Conv2d(image_shape[0], channels[0], kernels[0], strides[0], paddings[0]),
            self._activation(),
        )
        self.conv_layer2 = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernels[1], strides[1], paddings[1]),
            self._activation(),
        )
        self.conv_layer3 = nn.Sequential(
            nn.Conv2d(channels[1], channels[2], kernels[2], strides[2], paddings[2]),
            self._activation(),
        )
        self.downsample = nn.Sequential(
            nn.Linear(linear_size, 2 * latent_dim),
            self._activation(),
        )
        self.mean_layer = nn.Sequential(nn.Linear(2 * latent_dim, latent_dim))
        self.log_var_layer = nn.Sequential(
            nn.Linear(2 * latent_dim, latent_dim),
        )

    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.conv_layer3(out)
        out = out.view(out.size(0), -1)
        out = self.downsample(out)

        mean = self.mean_layer(out)
        log_var = self.log_var_layer(out)

        return mean, log_var


class VaeDecoder(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        latent_dim: int,
        activation: nn.Module = nn.Mish,
    ):
        super().__init__()

        self._image_shape = image_shape
        self._latent_dim = latent_dim
        self._activation = activation

        c, h, w = image_shape
        conv1_shape = conv_out_shape((h, w), paddings[0], kernels[0], strides[0])
        conv2_shape = conv_out_shape(conv1_shape, paddings[1], kernels[1], strides[1])
        conv3_shape = conv_out_shape(conv2_shape, paddings[2], kernels[2], strides[2])
        self.conv_shape = (channels[0], *conv3_shape)  # デコーダCNN層に通す直前のshape

        # 以下最適化対象のモデル
        self.upsample_layer = nn.Linear(
            self._latent_dim, channels[0] * np.prod(conv3_shape).item()
        )
        self.conv_tras_layer1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels[0],
                out_channels=channels[1],
                kernel_size=kernels[0],
                stride=strides[0],
                padding=paddings[0],
            ),
            self._activation(),
        )
        self.conv_tras_layer2 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels[1],
                out_channels=channels[2],
                kernel_size=kernels[1],
                stride=strides[1],
                padding=paddings[1],
            ),
            self._activation(),
        )
        self.conv_tras_layer3 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=channels[2],
                out_channels=c,
                kernel_size=kernels[2],
                stride=strides[2],
                padding=paddings[2],
            )
        )

    def forward(self, x):
        """
        :param x: size(*batch_shape, embed_size)
        :return: obs_dist = size(*batch_shape, *self.shape)
        """
        batch_shape = x.shape[:-1]
        embed_size = x.shape[-1]
        squeezed_size = np.prod(batch_shape).item()
        x = x.reshape(squeezed_size, self.feature_size)

        x = self.upsample_layer(x)
        x = torch.reshape(x, (squeezed_size, *self.conv_shape))
        feature_map = self.conv_tras_layer1(x)
        feature_map = self.conv_tras_layer2(feature_map)
        reconst = self.conv_tras_layer3(feature_map)

        return reconst