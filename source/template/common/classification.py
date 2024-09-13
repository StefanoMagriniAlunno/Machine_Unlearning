from typing import Callable, Tuple

import torch


def batch_loss(
    classificator: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: torch.nn.Module,
    regularization: Callable[[torch.nn.Module], torch.tensor] | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    y_: torch.Tensor = classificator(x)

    # compute loss
    loss: torch.Tensor = loss_fn(y_, y)
    # regularization loss
    if regularization is not None:
        loss = loss + regularization(classificator)

    # compute mean confidence
    conf = y_.max(dim=1).values.mean()

    # compute accuracy
    attributions = (y_.argmax(dim=1) == y).sum()

    return loss, attributions, conf
