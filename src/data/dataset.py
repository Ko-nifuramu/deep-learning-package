import torch
import numpy as np


def create_goal_image_data(vision_target_data: np) -> np:
    if len(vision_target_data.shape) == 4:
        vision_goal_data = vision_target_data[-1:, :, :, :]
    else:
        vision_goal_data_1step = vision_target_data[:, -1:, :, :, :]
        for i in range(vision_target_data.shape[1]):
            if i == 0:
                vision_goal_data = vision_goal_data_1step
                continue
            vision_goal_data = np.concatenate(
                [vision_goal_data, vision_goal_data_1step], axis=1
            )

    return vision_goal_data


class ManipulatorDataSet(torch.utils.data.Dataset):
    def __init__(
        self,
        vision_input_data: np,
        vision_target_data,
        joint_input_data: np,
        joint_target_data: np,
        image_noise_std: float,
        joint_noise_std: float,
    ):

        self.v_input_data = torch.from_numpy(vision_input_data).float()
        self.v_target_data = torch.from_numpy(vision_target_data).float()
        self.j_input_data = torch.from_numpy(joint_input_data).float()
        self.j_target_data = torch.from_numpy(joint_target_data).float()
        self.v_goal_data = torch.from_numpy(
            create_goal_image_data(vision_target_data)
        ).float()

        self.image_noise_std = image_noise_std
        self.joint_data_noise_std = joint_noise_std

    def __len__(self):
        return self.v_input_data.shape[0]

    def __getitem__(self, index):
        # ノイズを加える
        vision_input_datum = self.v_input_data[index] + torch.normal(
            mean=0, std=self.image_noise_std, size=self.v_input_data[index].shape
        )
        vision_target_datum = self.v_target_data[index] + torch.normal(
            mean=0, std=self.image_noise_std, size=self.v_target_data[index].shape
        )
        joint_input_datum = self.j_input_data[index] + torch.normal(
            mean=0, std=self.joint_data_noise_std, size=self.j_input_data[index].shape
        )
        joint_target_datum = self.j_target_data[index] + torch.normal(
            mean=0, std=self.joint_data_noise_std, size=self.j_target_data[index].shape
        )
        vision_goal_datum = self.v_goal_data[index] + torch.normal(
            mean=0, std=self.joint_data_noise_std, size=self.v_goal_data[index].shape
        )

        # 元の地域に戻す
        vision_input_datum = torch.clamp(vision_input_datum, min=0, max=1)
        joint_input_datum = torch.clamp(
            joint_input_datum, min=-0.95, max=0.95
        )  # -1, 1付近も含めてしまうと活性化関数tanhで勾配消失が起きてしまう
        joint_target_datum = torch.clamp(joint_target_datum, min=-0.95, max=0.95)

        return (
            vision_input_datum,
            vision_target_datum,
            joint_input_datum,
            joint_target_datum,
            vision_goal_datum,
        )
