from itertools import product

import matplotlib.pyplot as plt


def visualize_similarities(square_size: int, image_path_list: list[str]) -> None:

    fig, axs = plt.subplots(nrows=square_size, ncols=square_size)
    for i, (col, row) in enumerate(product(range(square_size), range(square_size))):
        axs[col, row].imshow(plt.imread(image_path_list[i]))

    for ax in axs.flat:
        ax.set_axis_off()
