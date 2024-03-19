import torch.distributions as td
import torch.nn as nn
import yaml

from src.agent import VaeRnnAgent
from src.models.rnn import GRU
from src.models.vae import VAE, VaeDecoder, VaeEncoder

from src.utils.data_utils import get_device


# 正直、環境変数に読み込んだほうがいいかも
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def rnn_builder(config_path: str):
    config = load_config(config_path)

    rnn_config = config["rnn"]
    rnn_hidden_dim = rnn_config["parameter"]["rnn_hidden_dim"]
    num_layers = rnn_config["parameter"]["num_layers"]
    input_dim = rnn_config["parameter"]["input_dim"]
    latent_dim = rnn_config["parameter"]["latent_dim"]
    joint_dim = rnn_config["parameter"]["joint_dim"]
    activation = getattr(nn, rnn_config["activation"])

    mean_hidden_dims = rnn_config["parameter"]["mean_hidden_dims"]
    log_var_hidden_dims = rnn_config["parameter"]["log_var_hidden_dims"]
    joint_hidden_dims = rnn_config["parameter"]["joint_hidden_dims"]

    rnn_model = GRU(
        latent_dim=latent_dim,
        joint_dim=joint_dim,
        input_dim=input_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        num_layers=num_layers,
        mean_hidden_dims=mean_hidden_dims,
        log_var_hidden_dims=log_var_hidden_dims,
        joint_hidden_dims=joint_hidden_dims,
        activation=activation,
    )

    return rnn_model


def vae_builder(config_path: str):
    config = load_config(config_path)

    encoder_params = config["vae"]["parameter"]["encoder"]
    decoder_params = config["vae"]["parameter"]["decoder"]

    latent_dim = config["vae"]["latent_dim"]

    encoder = VaeEncoder(
        obs_shape=encoder_params["obs_shape"],
        channels=encoder_params["channels"],
        kernels=encoder_params["kernels"],
        strides=encoder_params["strides"],
        paddings=encoder_params["paddings"],
        latent_dim=latent_dim,
        activation=getattr(nn, config['vae']['conv_activation']),
    )

    decoder = VaeDecoder(
        obs_shape=decoder_params["obs_shape"],
        channels=decoder_params["channels"],
        kernels=decoder_params["kernels"],
        strides=decoder_params["strides"],
        paddings=decoder_params["paddings"],
        latent_dim=latent_dim,
        conv_activation=getattr(nn, config['vae']['conv_activation']),
        reconst_activation=getattr(nn, config['vae']['reconst_activation']),
    )

    return VAE(get_device(), latent_dim, encoder, decoder)


def rnn_vae_agent_model_builder(config_path: str):
    vae_model = vae_builder(config_path)
    rnn_model = rnn_builder(config_path)

    config = load_config(config_path)

    agent_model = VaeRnnAgent(
        obs_shape=config['dataset']["obs_shape"],
        joint_dim=config['dataset']["joint_dim"],
        latent_dim=config['vae']["latent_dim"],
        vae_model=vae_model,
        rnn_model=rnn_model,
        kld_weight=config['optimize_setting']["kld_weight"],
        vae_weight=config['optimize_setting']["vae_weight"],
        joint_weight=config['optimize_setting']["joint_weight"],
    )
    return agent_model
