"""Exploring parameterizations for convex set boundaries.."""

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.optim as optim
from tomsutils.structs import Image
from tomsutils.utils import fig2data
import imageio.v2 as iio


class SimpleNN(nn.Module):

    def __init__(self, input_size, hidden_layers, output_size):
        super(SimpleNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# NOTE: this is not even really convex, so what property do we need exactly?
def _generate_2d_data(num_data=1000) -> NDArray[np.float64]:
    t = np.linspace(0, 2 * np.pi, num_data)
    x = np.sin(t) + 0.5 * np.sin(t)
    y = 3 * np.cos(t) - 0.5 * np.cos(5 * t)
    return np.vstack([x, y]).T


def _create_visualization(
    model: SimpleNN,
    data: NDArray[np.float64],
    origin: tuple[float, float],
    eval_angles: NDArray[np.float64],
    suptitle: str,
) -> Image:
    # Get predictions.
    pred_distances = (
        model(torch.from_numpy(eval_angles[:, np.newaxis].astype(np.float32)))
        .detach()
        .numpy()
        .squeeze()
    )
    # Convert back into Cartesian coordinates for visualization.
    pred_x = pred_distances * np.cos(eval_angles) + origin[0]
    pred_y = pred_distances * np.sin(eval_angles) + origin[1]
    prediction = np.vstack([pred_x, pred_y]).T
    pred_x, pred_y = prediction.T

    x, y = data.T

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
    axes[0].set_title("Training Data")
    axes[0].scatter(x, y, color="purple", label="Data")
    axes[0].fill(x, y, color="pink", alpha=0.5)
    axes[0].scatter([origin[0]], [origin[1]], color="green", label="Origin")
    axes[0].legend(loc="lower right")
    axes[0].axis("off")
    axes[1].set_title("Learned Model")
    axes[1].scatter(pred_x, pred_y, color="blue", label="Prediction")
    axes[1].fill(x, y, color="pink", alpha=0.5)
    axes[1].scatter([origin[0]], [origin[1]], color="green", label="Origin")
    axes[1].legend(loc="lower right")
    axes[1].axis("off")
    plt.suptitle(suptitle)
    plt.tight_layout()
    img = fig2data(fig)
    plt.close()
    return img


def _run():
    # Generate data from some funky 2D convex shape.
    num_data = 100
    num_dims = 2
    data = _generate_2d_data(num_data)
    assert data.shape == (num_data, num_dims)

    # Assume that this is in the interior.
    origin = np.mean(data, axis=0)

    # Reparameterize the data in polar coordinates.
    angles = np.arctan2(data[:, 1] - origin[1], data[:, 0] - origin[0])
    distances = np.linalg.norm(data - origin, axis=1)

    # Initialize the model.
    output_size = 1  # distance to boundary
    input_size = num_dims - 1  # e.g., polar or spherical coordinates
    hidden_layers = [64, 64]
    model = SimpleNN(input_size, hidden_layers, output_size)

    # Predict distances from angles and add some more points not seen in training.
    num_eval = 5 * num_data
    eval_angles = np.linspace(-np.pi, np.pi, num_eval)

    # Define the loss function and the optimizer.
    learning_rate = 0.01
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop.
    num_epochs = 10000
    render_interval = num_epochs // 20
    training_inputs = torch.from_numpy(angles[:, np.newaxis].astype(np.float32))
    training_outputs = torch.from_numpy(distances[:, np.newaxis].astype(np.float32))
    imgs = []
    for epoch in range(num_epochs):

        if epoch % render_interval == 0:
            img = _create_visualization(
                model, data, origin, eval_angles, f"Epoch {epoch}"
            )
            imgs.append(img)

        # Forward pass.
        outputs = model(training_inputs)
        loss = criterion(outputs, training_outputs)

        # Backward pass and optimization.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    # Make a nice video.
    iio.mimsave("convex_set_boundaries_demo.mp4", imgs, fps=5)


if __name__ == "__main__":
    _run()
