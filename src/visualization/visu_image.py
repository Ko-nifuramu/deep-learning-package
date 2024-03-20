import matplotlib.animation as anime
import matplotlib.pyplot as plt
import numpy as np


def save_image(image: np.ndarray, save_folder_path: str, file_name: str):
    """
    image: min:0, max:1
    """
    save_path = save_folder_path + "/" + file_name
    plt.imshow(image)

    plt.savefig(save_path)


def gif_from_ndarray(image_data: np.ndarray, save_path: str, gif_name: str):
    gif_list = []
    print("-----making gif..... ------")
    print(f"input_image: {image_data.shape}")
    image_data = image_data[0]  # 一番目のデータをgifにする
    image_data = image_data.transpose(0, 2, 3, 1)
    print(f"gif_image: {image_data.shape}")

    fig = plt.figure()

    for i in range(image_data.shape[0]):
        img = [plt.imshow(image_data[i])]
        gif_list.append(img)

    ani = anime.ArtistAnimation(fig, gif_list, interval=100)
    ani.save(f"{save_path}/{gif_name}_double_spped.gif", writer="pillow")


def visualize_loss(
    train_loss_dict: dict,
    val_loss_dict: dict,
    folder_path: str,
    stop_epoch: int,
    fig_name: str = "",
):
    legend_list = []

    fig = plt.figure()
    ax = fig.add_subplot()
    for key, loss_list in train_loss_dict.items():
        loss_list = [item for item in loss_list]

        ax.plot(loss_list, linestyle="solid")
        legend_list.append(key)

    for key, loss_list in val_loss_dict.items():
        loss_list = [item for item in loss_list]

        ax.plot(loss_list, linestyle="dashed")
        legend_list.append(key)

    ax.set_yscale("log")
    ax.legend(legend_list, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xlim(0, stop_epoch)
    ax.set_title("each_loss")

    fig.savefig(
        "reports/figures/train/"
        + folder_path
        + "/"
        + "Training_loss_"
        + str(stop_epoch),
        bbox_inches="tight",
    )
