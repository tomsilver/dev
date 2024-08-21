import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import RangeSlider
from numpy.typing import NDArray

from dataset import DIMENSION_NAMES, load_subject_condition_data


def _main(data_dir: Path, subject_id: int, condition_name: str) -> None:
    arr = load_subject_condition_data(data_dir, subject_id, condition_name)
    title = f"{condition_name} [subject {subject_id}]"
    _create_4d_interactive_visualization(arr, title)


def _create_4d_interactive_visualization(
    data: NDArray,
    title: str,
    axcolor: str = "lightgoldenrodyellow",
    dim_labels: tuple[str, str, str, str] = DIMENSION_NAMES,
    scatter_color: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 0.25),
    scatter_size: float = 0.5,
) -> None:
    plt.rc("font", size=12)

    x, y, z, c = data.T

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()
    c_min, c_max = c.min(), c.max()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, s=scatter_size, color=scatter_color)  # type: ignore

    ax_slider = plt.axes((0.2, 0.02, 0.65, 0.03), facecolor=axcolor)
    slider = RangeSlider(ax_slider, dim_labels[3], c_min, c_max, valinit=(c_min, c_max))

    def update(val):
        nonlocal sc
        min_val, max_val = val
        mask = (c >= min_val) & (c <= max_val)
        sc.remove()
        sc = ax.scatter(x[mask], y[mask], z[mask], s=scatter_size, color=scatter_color)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    ax.set_xlim((x_min, x_max))
    ax.set_ylim((y_min, y_max))
    ax.set_zlim((z_min, z_max))  # type: ignore

    ax.set_xlabel(DIMENSION_NAMES[0])
    ax.set_ylabel(DIMENSION_NAMES[1])
    ax.set_zlabel(DIMENSION_NAMES[2])  # type: ignore

    plt.suptitle(title)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--subject_id", type=int, required=True)
    parser.add_argument("--condition", type=str, required=True)
    args = parser.parse_args()
    _main(args.data_dir, args.subject_id, args.condition)
