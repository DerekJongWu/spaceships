from typing import Tuple

import torch
from torch import Tensor


def modulated_loss(pred: Tensor, target: Tensor) -> Tensor:
    """5 parameter modulated rotation loss
    Arguments:
        pred {Tensor Batch} -- x, y, yaw, w, h
        target {Tensor Batch} -- x, y, yaw, w, h
        * X and Y position (centre of the bounding box)
        * Yaw (direction of heading)
        * Width (size tangential to the direction of yaw)
        * Height (size along the direct of yaw)
    Returns:
        loss for each pred, target pair without sum
    Reference: Eqn(2) https://arxiv.org/pdf/1911.08299.pdf
    """
    assert pred.shape[-1] == 5
    assert target.shape[-1] == 5

    x1, x2 = pred[:, 0], target[:, 0]
    y1, y2 = pred[:, 1], target[:, 1]
    yaw1, yaw2 = pred[:, 2], target[:, 2]
    w1, w2 = pred[:, 3], target[:, 3]
    h1, h2 = pred[:, 4], target[:, 4]

    # center point loss
    lcp = torch.abs(x1 - x2) + torch.abs(y1 - y2)

    lmr5p_ = torch.min(
        lcp + torch.abs(w1 - w2) + torch.abs(h1 - h2) + torch.abs(yaw1 - yaw2),
        lcp
        + torch.abs(w1 - h2)
        + torch.abs(h1 - w2)
        + torch.abs(90 - torch.abs(yaw1 - yaw2)),
    )

    return lmr5p_