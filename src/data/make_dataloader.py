# datalaoder
import numpy as np
from torch.utils.data import DataLoader

import src.data.data_processing as joint_regularization
import src.data.dataset as ManipulatorDataSet


def create_dataloader(
    image_data: np,
    joint_data: np,
    batch_size: int,
    image_noise_std: float,
    joint_noise_std: float,
    trainsample_ratio: float = 0.8,
) -> DataLoader:
    joint_data, joint_eachdim_maxmin_array = joint_regularization(joint_data)
    ratio = trainsample_ratio

    indeces = [int(image_data.shape[0] * n) for n in [ratio / 2, 0.5, (1 + ratio) / 2]]
    train_image_data1, val_image_data1, train_image_data2, val_image_data2 = np.split(
        image_data[:, 0:-1, :, :, :], indeces, axis=0
    )
    train_image_data = np.concatenate([train_image_data1, train_image_data2], axis=0)
    val_image_data = np.concatenate([val_image_data1, val_image_data2], axis=0)

    train_image_data1, val_image_data1, train_image_data2, val_image_data2 = np.split(
        image_data[:, 1:, :, :, :], indeces, axis=0
    )
    train_image_target_data = np.concatenate(
        [train_image_data1, train_image_data2], axis=0
    )
    val_image_target_data = np.concatenate([val_image_data1, val_image_data2], axis=0)

    train_joint_input1, val_joint_input1, train_joint_input2, val_joint_input2 = (
        np.split(joint_data[:, 0:-1, :], indeces, axis=0)
    )
    train_joint_input_data = np.concatenate(
        [train_joint_input1, train_joint_input2], axis=0
    )
    val_joint_input_data = np.concatenate([val_joint_input1, val_joint_input2], axis=0)

    train_joint_input1, val_joint_input1, train_joint_input2, val_joint_input2 = (
        np.split(joint_data[:, 1:, :], indeces, axis=0)
    )
    train_joint_target_data = np.concatenate(
        [train_joint_input1, train_joint_input2], axis=0
    )
    val_joint_target_data = np.concatenate([val_joint_input1, val_joint_input2], axis=0)

    train_dataset = ManipulatorDataSet(
        train_image_data,
        train_image_target_data,
        train_joint_input_data,
        train_joint_target_data,
        image_noise_std,
        joint_noise_std,
    )
    val_dataset = ManipulatorDataSet(
        val_image_data,
        val_image_target_data,
        val_joint_input_data,
        val_joint_target_data,
        image_noise_std,
        joint_noise_std,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    return train_dataloader, val_dataloader
