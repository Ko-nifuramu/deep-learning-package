import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.models.rnn import RNN
from src.models.vae import VAE


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

        self.z_var = self.vae.var
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

    def _optimize_setup(self, config_path: str):
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        self._batch_size = config["dataset"]["batch_size"]
        self.train_time_step = config["dataset"]["train_time_step"]
        optimize_setting_config = config["optimize_setting"]

        self.reconst_weight = optimize_setting_config["world_model"]["reconst_weight"]
        self.embed_weight = optimize_setting_config["world_model"]["embed_weight"]
        self.lower_kl_weight = optimize_setting_config["world_model"]["lower_kl_weight"]

        self.higher_kl_weight = optimize_setting_config["world_model"][
            "higher_kl_weight"
        ]

        self.grad_clip = optimize_setting_config["world_model"]["grad_clip"]
        self.lr_world_model = optimize_setting_config["world_model"]["lr"]
        self.world_max_epochs = optimize_setting_config["world_model"]["epochs"]
        self.world_model_optimizer = getattr(
            optim, optimize_setting_config["world_model"]["optimizer"]
        )(self.world_model.parameters(), lr=self.lr_world_model)
        self.is_use_kl_balancing = optimize_setting_config["world_model"][
            "is_use_kl_balancing"
        ]
        self.kl_balancing_alpha = optimize_setting_config["world_model"][
            "kl_balancing_alpha"
        ]

    def cal_loss(
        self, i_input, i_target, j_input, j_target, i_g, criterion=nn.MSELoss()
    ):

        loss_dict = {"total": 0, "vae": 0, "joint": 0, "recon": 0, "kld": 0}
        scale_adjust = (
            i_input.shape[1] * i_input.shape[2] * i_input.shape[3] / self.z_dim
        )

        # vae encoder, reparameterize
        mean, log_var = self.vision_vae.encoder(i)
        z_i = self.vision_vae.reparameterize(mean, log_var)
        mean_g, log_var_g = self.vision_vae.encoder(i_g)
        z_i_g = self.vision_vae.reparameterize(mean_g, log_var_g)

        # RNN
        mean_hat, log_var_hat, j_out = self.rnn(z_i, z_i_g, j_input)
        z_latent_next = self.vision_vae.reparameterize(mean_hat, log_var_hat)

        j_next = torch.tanh(j_out)
        prediction_image_next = self.vision_vae.decoder(
            z_latent_next.reshape(-1, self.z_dim)
        )

        # calculate losses
        loss_joint = criterion(j_target, j_next)
        image_recon = criterion(i_target, prediction_image_next)

        kld = -torch.mean(1 + log_var - mean**2 - torch.exp(log_var)) / 2

        loss_vae = image_recon * scale_adjust / (self.var * 2) + self.beta * kld

        loss = self.weight_vae * loss_vae + self.weight_joint * loss_joint

        loss_dict["total"] = loss.copy().detach()
        loss_dict["vae"] = loss_vae.copy().detach()
        loss_dict["joint"] = loss_joint.copy().detach()
        loss_dict["recon"] = image_recon.copy().detach()
        loss_dict["kld"] = kld.copy().detach()

        return loss, loss_dict

    def fit(
        self,
        train_datalaoder: DataLoader,
        val_datalaoder: DataLoader,
        epochs: int,
        criterion=nn.MSELoss(),
    ):

        is_train_model = input("Do you want to train the model? (y/n): ")

        if is_train_model == "n":
            return None

        train_loss_dict = {
            "train_world": [],
            "train_reconst": [],
            "train_lower_kld": [],
            "train_higher_kld": [],
            "train_embed": [],
        }
        val_loss_dict = {
            "val_world": [],
            "val_reconst": [],
            "val_lower_kl": [],
            "val_higher_kl": [],
            "val_embed": [],
        }

        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

        train_loss_dict = {"total": [], "vae": [], "joint": [], "recon": [], "kld": []}
        val_loss_dict = {"total": [], "vae": [], "joint": [], "recon": [], "kld": []}

        for epoch in range(epochs):
            self.train()
            for i_input, i_target, j_pre, j_target, i_g in train_datalaoder:
                optimizer.zero_grad()
                loss, loss_dict = self.cal_loss(
                    i_input, i_target, j_pre, j_target, i_g, criterion
                )
                loss.backward()
                optimizer.step()
                train_loss_dict["total"].append(loss_dict["total"])
                train_loss_dict["vae"].append(loss_dict["vae"])
                train_loss_dict["joint"].append(loss_dict["joint"])
                train_loss_dict["recon"].append(loss_dict["recon"])
                train_loss_dict["kld"].append(loss_dict["kld"])

            self.eval()
            with torch.no_grad():
                for i_input, i_target, j_pre, j_target, i_g in val_datalaoder:
                    loss, loss_dict = self.cal_loss(
                        i_input, i_target, j_pre, j_target, i_g, criterion
                    )
                    val_loss_dict["total"].append(loss_dict["total"])
                    val_loss_dict["vae"].append(loss_dict["vae"])
                    val_loss_dict["joint"].append(loss_dict["joint"])
                    val_loss_dict["recon"].append(loss_dict["recon"])
                    val_loss_dict["kld"].append(loss_dict["kld"])

            scheduler.step()

        return train_loss_dict, val_loss_dict
