import torch.distributions as td
import torch.nn as nn
import yaml


# 正直、環境変数に読み込んだほうがいいかも
def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def world_model_builder(config_path: str):
    config = load_config(config_path)

    world_config = config["world_model"]
    embed_size = world_config["embed_obs_dim"]

    rssm_config = world_config["rssm"]

    train_time_step = config["dataset"]["train_time_step"]

    rssm_model = RSSMRepresentation(
        obs_embed_size=embed_size,
        action_size=world_config["action_size"],
        stoch_size=rssm_config["parameter"]["stoch_size"],
        deter_size=rssm_config["parameter"]["deter_size"],
        hidden_size=rssm_config["parameter"]["hidden_size"],
        categorical_classes=rssm_config["parameter"]["categorical_classes"],
        softmax_temperature=rssm_config["parameter"]["softmax_temperature"],
        _activation=getattr(nn, rssm_config["activation"]),
        dist_name=rssm_config["stoch_dist_name"],
        rnn=getattr(nn, rssm_config["rnn"]),
    )

    encoder_config = world_config["vae"]["parameter"]["encoder"]

    image_shape = tuple(encoder_config["obs_shape"])
    obs_encoder_model = ObserEncoder(
        image_shape=image_shape,
        channels=tuple(encoder_config["channels"]),
        kernels=tuple(encoder_config["kernels"]),
        strides=tuple(encoder_config["strides"]),
        paddings=tuple(encoder_config["paddings"]),
        embed_size=world_config["embed_obs_dim"],
        activation=getattr(nn, world_config["vae"]["conv_activation"]),
    )

    decoder_config = world_config["vae"]["parameter"]["decoder"]

    obs_decoder_model = ObserDecoder(
        image_shape=image_shape,
        channels=tuple(decoder_config["channels"]),
        kernels=tuple(decoder_config["kernels"]),
        strides=tuple(decoder_config["strides"]),
        paddings=tuple(decoder_config["paddings"]),
        dist_std=decoder_config["dist_std"],
        feature_size=rssm_config["parameter"]["stoch_size"]
        + rssm_config["parameter"]["deter_size"],
        embed_size=world_config["embed_obs_dim"],
        activation=getattr(nn, world_config["vae"]["conv_activation"]),
    )

    return WorldModel(
        image_shape,
        obs_encoder_model,
        obs_decoder_model,
        rssm_model,
        train_time_step=train_time_step,
    )
