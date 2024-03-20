import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.rnn import GRU as RNN
from src.models.vae import VAE
from src.visualization.visu_image import gif_from_ndarray


class VaeRnnAgent(nn.Module):
    def __init__(
        self,
        obs_shape: tuple,
        joint_dim: int,
        latent_dim: int,
        vae_model: VAE,
        rnn_model: RNN,
        kld_weight: float,
        vae_weight: float,
        joint_weight: float,
    ):
        super(VaeRnnAgent, self).__init__()

        self.obs_shape = obs_shape
        self.joint_dim = joint_dim
        self.latent_dim = latent_dim

        self.vision_vae = vae_model
        self.rnn = rnn_model

        self.z_var = vae_model.var
        self.beta = kld_weight
        self.weight_vae = vae_weight
        self.weight_joint = joint_weight

    def forward(self, i_pre, j_pre, i_g):
        z_i_pre = self.vision_vae.forward_z(i_pre)
        z_i_g = self.vision_vae.forward_z(i_g)
        mean_hat, log_var_hat, j_next = self.rnn(z_i_pre, z_i_g, j_pre)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_next)
        recon_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.latent_dim)
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
            z_latent_next.reshape(-1, self.latent_dim)
        )

        return j_next, recon_image_next, h

    def cal_loss(
        self,
        i_input,
        i_target,
        j_input,
        j_target,
        i_g,
        criterion=nn.MSELoss(),
        flag=False,
    ):

        if flag:
            gif_from_ndarray(
                i_input.clone().cpu().detach().numpy(), "reports/figures", "input_image"
            )

        loss_dict = {"total": 0, "vae": 0, "joint": 0, "recon": 0, "kld": 0}
        scale_adjust = (
            i_input.shape[1] * i_input.shape[2] * i_input.shape[3] / self.latent_dim
        )

        # vae encoder, reparameterize
        mean, log_var = self.vision_vae.encoder(i_input)
        z_i = self.vision_vae.reparameterize(mean, log_var)
        mean_g, log_var_g = self.vision_vae.encoder(i_g)
        z_i_g = self.vision_vae.reparameterize(mean_g, log_var_g)

        # RNN
        mean_hat, log_var_hat, j_out = self.rnn(z_i, z_i_g, j_input)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_out)
        prediction_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.latent_dim)
        ).reshape(i_input.shape)

        # calculate losses
        loss_joint = criterion(j_target, j_next)
        image_recon = criterion(i_target, prediction_image_next)

        kld = -torch.mean(1 + log_var - mean**2 - torch.exp(log_var)) / 2

        loss_vae = (
            image_recon * scale_adjust / (self.vision_vae.var * 2) + self.beta * kld
        )

        loss = self.weight_vae * loss_vae + self.weight_joint * loss_joint

        loss_dict["total"] = loss.clone().detach()
        loss_dict["vae"] = loss_vae.clone().detach()
        loss_dict["joint"] = loss_joint.clone().detach()
        loss_dict["recon"] = image_recon.clone().detach()
        loss_dict["kld"] = kld.clone().detach()

        if flag:
            gif_from_ndarray(
                prediction_image_next.clone().cpu().detach().numpy(),
                "reports/figures",
                "prediction_next_image",
            )

        return loss, loss_dict

    def fit(
        self,
        train_datalaoder: DataLoader,
        val_datalaoder: DataLoader,
        epochs: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion=nn.MSELoss(),
    ):

        is_train_model = input("Do you want to train the model? (y/n): ")

        if is_train_model != "y":
            return None

        train_loss_dict = {
            "train_total": [],
            "train_vae": [],
            "train_joint": [],
            "train_recon": [],
            "train_kld": [],
        }
        val_loss_dict = {
            "val_total": [],
            "val_vae": [],
            "val_joint": [],
            "val_recon": [],
            "val_kld": [],
        }
        flag = False
        for epoch in tqdm(range(epochs)):
            self.train()
            for i_input, i_target, j_pre, j_target, i_g in train_datalaoder:
                optimizer.zero_grad()
                loss, loss_dict = self.cal_loss(
                    i_input, i_target, j_pre, j_target, i_g, criterion, flag
                )
                loss.backward()
                optimizer.step()
                train_loss_dict["train_total"].append(loss_dict["total"])
                train_loss_dict["train_vae"].append(loss_dict["vae"])
                train_loss_dict["train_joint"].append(loss_dict["joint"])
                train_loss_dict["train_recon"].append(loss_dict["recon"])
                train_loss_dict["train_kld"].append(loss_dict["kld"])

                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                if epoch == epochs - 2:
                    flag = True

            self.eval()
            with torch.no_grad():
                for i_input, i_target, j_pre, j_target, i_g in val_datalaoder:
                    loss, loss_dict = self.cal_loss(
                        i_input, i_target, j_pre, j_target, i_g, criterion
                    )
                    val_loss_dict["val_total"].append(loss_dict["total"])
                    val_loss_dict["val_vae"].append(loss_dict["vae"])
                    val_loss_dict["val_joint"].append(loss_dict["joint"])
                    val_loss_dict["val_recon"].append(loss_dict["recon"])
                    val_loss_dict["val_kld"].append(loss_dict["kld"])

        return train_loss_dict, val_loss_dict, epoch + 1
