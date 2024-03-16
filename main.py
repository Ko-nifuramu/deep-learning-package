import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.models import VaeRnnAgent
from src.visualization import visualize_loss


def train_agent(
    agent: VaeRnnAgent,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    model_name: str,
    device: torch.device,
):
    print(f"=========================\n agent training....\n=========================")

    agent.to(device)
    agent.train()

    train_loss_dict, val_loss_dict, stop_epoch = agent.fit(
        train_dataloader, val_dataloader
    )

    folder_path = "model/" + model_name
    another_info = "stopepoch" + str(stop_epoch)

    vae_model, rnn_model = agent.vae, agent.rnn
    torch.save(vae_model.state_dict(), folder_path + "/vae_" + another_info)
    torch.save(rnn_model.state_dict(), folder_path + "/rnn_" + another_info)

    visualize_loss(train_loss_dict, val_loss_dict, model_name, stop_epoch)


def test_agent(config_name: str, config_folder_path: str):
    config_path = config_folder_path + config_name + ".yaml"

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    image_data: np.ndarray = np.load(config["data_path"]["image_data"])  # uint8
    joint_data: np.ndarray = np.load(config["data_path"]["joint_data"])

    # play_image_data = play_image_data.transpose(0, 1, 4,2,3)

    agent_model = VaeRnnAgent(model_name=config_name, config_path=config_path)
    agent_model.vae.load_state_dict(
        torch.load(
            "data/saved_model/world_model/"
            + config_name
            + "/stopepoch7500_mtrssmWorldModel.pth"
        )
    )


def main():
    is_train_agent = True
    is_test_agent = True

    config_agent_name_list = ["example"]

    for config_name in config_agent_name_list:
        config_folder_path = "config/model/agent_model/"
        if is_train_agent:
            train_agent(config_name, config_folder_path)
        if is_test_agent:
            test_agent(config_name, config_folder_path)

    return 0


if __name__ == "__main__":
    main()
