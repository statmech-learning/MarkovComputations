"""
Training functions for Markov ICL regression models.
"""

import torch
import numpy as np


def train_model_regression(
    model,
    train_loader,
    val_loader,
    device,
    n_epochs=200,
    lr=0.001,
    method="direct_solve",
):
    """
    Train regression model with MSE loss.

    Returns:
        dict: training history
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    history = {
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
    }

    for epoch in range(n_epochs):
        model.train()
        train_losses = []
        train_abs_errors = []

        for z_seq, y_targets in train_loader:
            z_seq = z_seq.to(device)
            y_targets = y_targets.to(device).float()

            optimizer.zero_grad()
            y_pred = model(z_seq, method=method)
            loss = criterion(y_pred, y_targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            train_abs_errors.append(torch.abs(y_pred - y_targets).mean().item())

        model.eval()
        val_losses = []
        val_abs_errors = []

        with torch.no_grad():
            for z_seq, y_targets in val_loader:
                z_seq = z_seq.to(device)
                y_targets = y_targets.to(device).float()
                y_pred = model(z_seq, method=method)
                val_losses.append(criterion(y_pred, y_targets).item())
                val_abs_errors.append(torch.abs(y_pred - y_targets).mean().item())

        history["train_mse"].append(np.mean(train_losses))
        history["val_mse"].append(np.mean(val_losses))
        history["train_mae"].append(np.mean(train_abs_errors))
        history["val_mae"].append(np.mean(val_abs_errors))

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1:3d} | "
                f"Train MSE: {history['train_mse'][-1]:.6f} | "
                f"Val MSE: {history['val_mse'][-1]:.6f} | "
                f"Val MAE: {history['val_mae'][-1]:.6f}"
            )

    return history
