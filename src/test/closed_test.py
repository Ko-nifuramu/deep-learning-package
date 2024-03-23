import torch

from src.visualization.visu_image import gif_from_ndarray
from src.visualization.visu_joint import visualize_joint_test


# closed-test -> image : open, joint : closed
def closed_test(
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

    next_joint_predictions = torch.zeros_like(target_joints)
    print("next_joint_predictions.shape:", next_joint_predictions.shape)
    next_reconst_images = torch.zeros_like(target_images)
    print("next_reconst_images.shape:", next_reconst_images.shape)

    first_joint = input_joints[:, 0:1, :]

    joint_predictions = first_joint
    h = None

    for time in range(input_images.shape[1]):
        print("time:", time)
        joint_predictions, next_images, h = agent.test_step(
            input_images[:, time].reshape(-1, *image_shape),
            joint_predictions,
            goal_images[:, time].reshape(-1, *image_shape),
            h,
        )
        next_joint_predictions[:, time : time + 1] = joint_predictions
        next_reconst_images[:, time] = next_images

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
