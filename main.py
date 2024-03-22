import numpy as np
import torch

# import torch_optimizer as optim<- AdaBeliefとか使いたい時に使う
import torch.optim as optim
import yaml
from torchinfo import summary

from src.agent import VaeRnnAgent
from src.data.make_dataloader import create_dataloader
from src.utils.data_utils import get_device, mkdir, print_np_data_info, torch_fix_seed
from src.utils.model_builder import rnn_vae_agent_model_builder
from src.visualization.visu_loss import visualize_loss
from src.test.open_test import open_test


def train_agent(config_name: str, config_folder_path: str):
    config_path = config_folder_path + config_name + ".yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    image_data: np.ndarray = (
        np.load(config["data_path"]["image_data"]).transpose(0, 1, 4, 2, 3) / 255
    )
    joint_data: np.ndarray = np.load(config["data_path"]["joint_data"])
    print_np_data_info(image_data, "image_data")
    print_np_data_info(joint_data, "joint_data")

    dataset_params = config["dataset"]
    train_dataloader, val_dataloader = create_dataloader(
        image_data,
        joint_data,
        dataset_params["batch_size"],
        dataset_params["image_noise_std"],
        dataset_params["joint_noise_std"],
        dataset_params["trainsample_ratio"],
    )

    print(f"=========================\n agent training....\n=========================")

    agent = rnn_vae_agent_model_builder(config_path)
    agent.to(get_device())

    summary(agent, [(5, 3, 32, 32), (5, 5), (5, 3, 32, 32)])
    agent.train()

    optimize_setting_config = config["optimize_setting"]
    optimizer = getattr(optim, optimize_setting_config["optimizer"])(
        agent.parameters(), lr=optimize_setting_config["lr"]
    )

    scheduler = None
    if "scheduler" in optimize_setting_config:
        scheduler = getattr(
            optim.lr_scheduler,
            optimize_setting_config["scheduler"]["name"],
        )(optimizer, **optimize_setting_config["scheduler"]["params"])

    train_loss_dict, val_loss_dict, stop_epoch = agent.fit(
        train_dataloader,
        val_dataloader,
        epochs=optimize_setting_config["epochs"],
        optimizer=optimizer,
        scheduler=scheduler,
    )

    folder_path = "models/" + config_name
    another_info = "stopepoch" + str(stop_epoch)

    mkdir(folder_path)
    mkdir("reports/figures/" + config_name)
    vae_model, rnn_model = agent.vision_vae, agent.rnn
    torch.save(vae_model.state_dict(), folder_path + "/vae_" + another_info)
    torch.save(rnn_model.state_dict(), folder_path + "/rnn_" + another_info)

    visualize_loss(
        train_loss_dict, val_loss_dict, stop_epoch, "reports/figures/" + config_name
    )


def test_agent(config_name: str, config_folder_path: str):
    config_path = config_folder_path + config_name + ".yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    image_data: np.ndarray = np.load(config["data_path"]["image_data"])  # uint8
    joint_data: np.ndarray = np.load(config["data_path"]["joint_data"])
    agent_model = VaeRnnAgent(model_name=config_name, config_path=config_path)
    agent_model.vae.load_state_dict(
        torch.load(
            "data/saved_model/world_model/"
            + config_name
            + "/stopepoch7500_mtrssmWorldModel.pth"
        )
    )


def main():
    torch_fix_seed(42)

    is_train_agent = True
    is_test_agent = False

    config_agent_name_list = ["example"]

    for config_name in config_agent_name_list:
        config_folder_path = "config/models/"
        if is_train_agent:
            train_agent(config_name, config_folder_path)
        if is_test_agent:
            test_agent(config_name, config_folder_path)

    return 0


if __name__ == "__main__":
    main()
