import torch
import torch.nn as nn
from src.models.vae import VAE
from src.models.rnn import RNN


class VaeRnnAgent(nn.Module):
    def __init__(
        self,
        image_shape: tuple,
        joint_dim: int,
        z_dim: int,
        vae_model: VAE,
        rnn_model: RNN,
        kld_weight: float,
        kld_hat_weight: float,
        vae_weight: float,
        joint_weight: float,
    ):
        super(VaeRnnAgent, self).__init__()

        self.image_shape = image_shape
        self.joint_dim = joint_dim
        self.z_dim = z_dim

        self.vae = vae_model
        self.rnn = rnn_model

        self.var = self.vae.var
        self.beta = kld_weight
        self.beta_hat = kld_hat_weight
        self.weight_vae = vae_weight
        self.weight_joint = joint_weight

    def forward(self, i_pre, j_pre, i_g):
        z_i_pre = self.vision_vae.forward_z(i_pre)
        z_i_g = self.vision_vae.forward_z(i_g)
        mean_hat, log_var_hat, j_next = self.rnn(z_i_pre, z_i_g, j_pre)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_next)
        recon_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.z_dim)
        )

        return j_next, recon_image_next

    def test_step(self, i_pre, j_pre, i_g, h):
        z_i_pre = self.vision_vae.forward_z(i_pre)
        z_i_g = self.vision_vae.forward_z(i_g)
        mean_hat, log_var_hat, j_next, h = self.rnn.autoregress(
            z_i_pre, z_i_g, j_pre, h
        )
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_next)
        recon_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.z_dim)
        )

        return j_next, recon_image_next, h

    def cal_loss(self, i, i_target, j_pre, j_target, i_g, criterion=nn.MSELoss()):

        loss_list = []
        loss_noweight_list = []
        scale_adjust = i.shape[1] * i.shape[2] * i.shape[3] / self.z_dim

        # vae encoder, reparameterize
        mean, log_var = self.vision_vae.encoder(i)
        z_i = self.vision_vae.reparameterize(mean, log_var)
        mean_g, log_var_g = self.vision_vae.encoder(i_g)
        z_i_g = self.vision_vae.reparameterize(mean_g, log_var_g)

        # RNN
        mean_hat, log_var_hat, j_out = self.rnn(z_i, z_i_g, j_pre)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_out)
        prediction_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.z_dim)
        )

        # calculate losses
        loss_joint = criterion(j_target, j_next)
        image_recon = criterion(i_target, prediction_image_next)

        kld = -torch.mean(1 + log_var - mean**2 - torch.exp(log_var)) / 2

        mean_hat = mean_hat.reshape(-1, self.z_dim)
        log_var_hat = log_var_hat.reshape(-1, self.z_dim)
        kld_hat = (
            torch.mean(
                -1
                + (log_var_hat - log_var)
                + (torch.exp(log_var) + (mean - mean_hat) ** 2) / torch.exp(log_var_hat)
            )
            / 2
        )
        loss_vae = (
            image_recon * scale_adjust / (self.var * 2)
            + self.beta * kld
            + self.beta_hat * kld_hat
        )

        loss = self.weight_vae * loss_vae + self.weight_joint * loss_joint

        loss_list.extend(
            loss,
            self.weight_vae * loss_vae,
            self.weight_joint * loss_joint,
            image_recon * scale_adjust / (self.var * 2),
            self.beta * kld,
            self.beta_hat * kld_hat,
        )
        loss_noweight_list.extend(
            image_recon + kld + kld_hat + loss_joint,
            image_recon + kld + kld_hat,
            loss_joint,
            image_recon,
            kld,
            kld_hat,
        )

        return loss_list, loss_noweight_list
