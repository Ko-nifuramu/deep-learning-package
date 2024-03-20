import os
import random

import numpy as np
import torch


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError:
        print("this dirctory is already made or a wrong path is given.")


def print_np_data_info(data: np.ndarray, data_name: str) -> None:

    print("--------" + data_name + "--------")
    print(f"type(data) : {type(data)}")
    print(f"data.dtype : {data.dtype}")
    print(f"data.shape : {data.shape}")
    print(f"data_max : {np.max(data)}")
    print(f"data_min : {np.min(data)}")


def print_tensor_data_info(data: torch.Tensor, data_name: str) -> None:
    print("--------" + data_name + "--------")
    print(f"type(data) : {type(data)}")
    print(f"data.dtype : {data.dtype}")
    print(f"data.shape : {data.shape}")
    print(f"data_max : {torch.max(data)}")
    print(f"data_min : {torch.min(data)}")


def torch_fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    # DataLoaderのworkerの固定
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(gpu_id=-1):
    """
    使えるならGPUを使う
    """
    if gpu_id >= 0 and torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda", gpu_id)
    else:
        print("Using CPU")
        return torch.device("cpu")
