import torch
import torch.nn as nn
from .dataset.dateset import Tsr


def loss_quantile(
    pred_y: Tsr,
    tg: Tsr,
    quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
) -> Tsr:

    losses = []
    for idx, qtl in enumerate(quantiles):
        err = tg - pred_y[..., idx].unsqueeze(-1)  # (B, 1)
        losses.append(torch.max((qtl - 1) * err, qtl * err).unsqueeze(-1))  # (B, 1, 1)
    losses = torch.cat(losses, dim=2)
    loss = losses.sum(dim=(-2, -1)).mean()
    return loss


def loss_mse(pred_y: Tsr, tg: Tsr) -> Tsr:
    loss = nn.MSELoss()(pred_y, tg)
    return loss
