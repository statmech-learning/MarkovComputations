"""
Training functions for ICL models.

Provides a training loop with ICL/IWL tracking. The loop optionally shows
a tqdm progress bar that updates the current train/val/IWL/ICL accuracies
in place; pass show_progress=False to fall back to the original "print one
line every 10 epochs" behavior.
"""

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from evaluation import evaluate_iwl, evaluate_icl_novel  # , evaluate_icl_swap


def train_model(model, train_loader, val_loader, device, n_epochs=200, lr=0.001,
                method='direct_solve', temperature=1.0, gmm=None, N=None, B=1,
                L=None, exact_copy=True, eval_frequency=10, n_eval_samples=500,
                min_max_choice=None, unique_labels=False, show_progress=True):
    """Train the classification model with ICL/IWL tracking.

    Args:
        model: ICL model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: torch device
        n_epochs: Number of training epochs
        lr: Learning rate
        method: Method for steady state computation (model-specific)
        temperature: Softmax temperature
        gmm: GaussianMixtureModel instance (needed for ICL/IWL evaluation)
        N: Number of context examples
        B: Burstiness parameter
        L: Number of output classes
        exact_copy: Whether query is exact copy of context item
        eval_frequency: How often to evaluate ICL/IWL (in epochs)
        n_eval_samples: Number of samples for each ICL/IWL evaluation
        min_max_choice: Optional min/max constraint passed through to data gen
        unique_labels: If True, ensure context labels are unique
        show_progress: If True, show a tqdm progress bar with live-updating
            train/val/IWL/ICL accuracies. If False, fall back to printing
            one line every 10 epochs.

    Returns:
        dict: Training history with losses, accuracies, and ICL/IWL metrics.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.NLLLoss()  # model emits log-probabilities

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'iwl_acc': [],   # In-Weight Learning
        'icl_acc': [],   # ICL primary: novel classes
        # 'icl_swap_acc': [],  # ICL secondary: label swapping (disabled)
    }

    # Most-recent eval values, kept across non-eval epochs so the postfix
    # can keep showing them.
    iwl_acc = None
    icl_novel_acc = None

    epoch_iter = range(n_epochs)
    if show_progress:
        epoch_iter = tqdm(
            epoch_iter, desc="Training", unit="epoch", dynamic_ncols=True
        )

    for epoch in epoch_iter:
        # === Training phase ===
        model.train()
        train_losses = []
        train_correct = 0
        train_total = 0

        for z_seq, labels, targets in train_loader:
            z_seq = z_seq.to(device)
            labels = labels.to(device)
            targets = targets.to(device).long() - 1  # 1-indexed -> 0-indexed

            optimizer.zero_grad()
            logits = model(z_seq, labels, method, temperature)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_losses.append(loss.item())
            preds = logits.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        # === Validation phase ===
        model.eval()
        val_losses = []
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for z_seq, labels, targets in val_loader:
                z_seq = z_seq.to(device)
                labels = labels.to(device)
                targets = targets.to(device).long() - 1

                logits = model(z_seq, labels, method, temperature)
                loss = criterion(logits, targets)

                val_losses.append(loss.item())
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total

        history['train_loss'].append(np.mean(train_losses))
        history['val_loss'].append(np.mean(val_losses))
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        # === Periodic ICL / IWL evaluation ===
        if gmm is not None and N is not None and (epoch + 1) % eval_frequency == 0:
            model.eval()
            iwl_acc = evaluate_iwl(
                model, gmm, N, device, n_eval_samples,
                L=L, method=method, temperature=temperature,
            )
            history['iwl_acc'].append(iwl_acc)

            icl_novel_acc = evaluate_icl_novel(
                model, gmm, N, device, n_eval_samples, exact_copy,
                B=B, L=L, method=method, temperature=temperature,
                min_max_choice=min_max_choice, unique_labels=unique_labels,
            )
            history['icl_acc'].append(icl_novel_acc)
        else:
            history['iwl_acc'].append(None)
            history['icl_acc'].append(None)

        # === Progress reporting ===
        if show_progress:
            postfix = {
                'train': f"{train_acc:.1f}%",
                'val': f"{val_acc:.1f}%",
            }
            if iwl_acc is not None:
                postfix['IWL'] = f"{iwl_acc:.1f}%"
            if icl_novel_acc is not None:
                postfix['ICL'] = f"{icl_novel_acc:.1f}%"
            epoch_iter.set_postfix(postfix)
        elif (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1:3d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%"
            if iwl_acc is not None:
                msg += f" | IWL: {iwl_acc:.2f}% | ICL: {icl_novel_acc:.2f}%"
            print(msg)

    return history
