import matplotlib.pyplot as plt
import numpy as np


def cal_MSE(joint_targets: np, joint_predictions: np) -> float:
    joint_targets = joint_targets.reshape(-1, 1)
    joint_predictions = joint_predictions.reshape(-1, 1)
    loss = 0

    for i in range(joint_predictions.shape[0]):
        loss = (joint_predictions[i, 0] - joint_targets[i, 0]) ** 2

    return loss / joint_predictions.shape[0]


def cal_max_differency(joint_targets: np, joint_predictions: np) -> float:
    diff_prediction_target = np.abs(joint_predictions - joint_targets)

    return np.max(diff_prediction_target)


def visualize_joint_test(
    joint_predictions: np, joint_target_data: np, report_path: str
):
    joint_predictions = joint_predictions.detach().cpu().numpy()
    joint_target_data = joint_target_data.detach().cpu().numpy()

    data_num = joint_predictions.shape[0] if len(joint_predictions.shape) == 3 else 1

    joint_dim = joint_predictions.shape[-1]

    legend_label = []
    for i in range(data_num):
        legend_label.append("pred_data{}".format(i + 1))
        legend_label.append("target_data{}".format(i + 1))

    mean_mse_loss = 0
    fig = plt.figure(figsize=(20, 10))

    plt.subplots_adjust(hspace=1, wspace=0.6)
    for i in range(data_num):
        for j in range(joint_dim):
            ax = fig.add_subplot(3, 2, j + 1)
            ax.set_xlim(1, 50)
            if i % 2 == 0:
                pre_color = "royalblue"
                target_color = "deepskyblue"
            else:
                pre_color = "seagreen"
                target_color = "mediumspringgreen"
            ax.plot(joint_predictions[i, :, j], "-", color=pre_color)
            ax.plot(joint_target_data[i, :, j], ":", color=target_color)
            ax.legend(legend_label, loc="center left", bbox_to_anchor=(1.0, 0.5))

            mean_mse_loss += cal_MSE(
                joint_target_data[i, :, j], joint_predictions[i, :, j]
            )
    plt.suptitle(
        f"compare_predictions_targets(mean_mse_loss : {mean_mse_loss/(5*data_num)})"
    )
    plt.savefig(report_path + "_compare_predictions_targets_data")
