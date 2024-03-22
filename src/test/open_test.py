import numpy as np
import torch

from src.data.data_processing import create_goal_image_data
from src.visualization.visu_joint import visualize_joint_test
from src.visualization.visu_image import gif_from_ndarray

# open-test -> image : open, joint : open
def open_test(
    agent,
    image_data: np.ndarray,
    joint_data: np.ndarray,
    report_path: str,
    data_name: str,
):
    image_shape = image_data.shape[-3:]

    joint_input_data = inputs_joint[:, 0:-1, :]
    joint_target_data = inputs_joint[:, 1:, :]
    image_inputs_data = inputs_image[:, 0:-1, :, :, :]

    # create_goal_image_data is (np->np)_function
    image_goal = create_goal_image_data(inputs_image[:, 1:, :, :, :])

    joint_input_data = torch.from_numpy(joint_input_data).float()
    joint_target_data = torch.from_numpy(joint_target_data).float()
    image_inputs_data = torch.from_numpy(image_inputs_data).float()
    image_goal = torch.from_numpy(image_goal).float()

    joint_predictions, reconst_images = agent.forward(
        image_inputs_data.reshape(-1, *image_shape),
        joint_input_data,
        image_goal.reshape(-1, *image_shape),
    )
    
    gif_from_ndarray(image_inputs_data.detach().numpy(), "reports/figures/example/open_test", "input_images")
    gif_from_ndarray(reconst_images.detach().numpy(),  "reports/figures/example/open_test", "reconst_images")

    visualize_joint_test(
        joint_predictions,
        joint_target_data,
        data_num_in_1image,
        report_path + "/open_" + data_name,
    )
