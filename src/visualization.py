from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt


def visualize_similarities(square_size: int, image_path_list: list[Path]) -> None:
    fig, axs = plt.subplots(nrows=square_size, ncols=square_size)
    for i, (col, row) in enumerate(product(range(square_size), range(square_size))):
        axs[col, row].imshow(plt.imread(image_path_list[i]))

    axs[0, 0].set_title("The original", fontsize=10)
    for ax in axs.flat:
        ax.set_axis_off()
    fig.suptitle("An original picture and its neighbours")
    plt.show(block=True)
