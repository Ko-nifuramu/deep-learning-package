import torch

from src.visualization.visu_image import gif_from_ndarray
from src.visualization.visu_joint import visualize_joint_test


# open-test -> image : open, joint : open
def open_test(
    agent,
    input_images: torch.Tensor,
    target_images: torch.Tensor,
    input_joints: torch.Tensor,
    target_joints: torch.Tensor,
    goal_images: torch.Tensor,
    report_path: str,
    data_name: str,
):

    image_shape = input_images.shape[-3:]
    len_time = input_images.shape[1]

    next_joint_predictions, next_reconst_images, _, _ = agent.forward(
        input_images.reshape(-1, *image_shape),
        input_joints,
        goal_images.reshape(-1, *image_shape),
    )

    print("target_images.shape:", target_images.shape)
    print("next_reconst_images.shape:", next_reconst_images.shape)
    next_reconst_images = next_reconst_images.reshape(-1, len_time, *image_shape)

    gif_from_ndarray(
        target_images.detach().numpy(), report_path, data_name + "_target_images"
    )
    gif_from_ndarray(
        next_reconst_images.detach().numpy(), report_path, data_name + "_reconst_images"
    )

    visualize_joint_test(
        next_joint_predictions,
        target_joints,
        report_path + data_name,
    )
