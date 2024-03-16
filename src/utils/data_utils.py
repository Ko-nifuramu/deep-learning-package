import numpy as np


def print_shape_and_maxMin(data: np) -> None:
    print(f"data.shape : {data.shape}")
    print(f"data_max : {np.max(data)}")
    print(f"data_min : {np.min(data)}")
