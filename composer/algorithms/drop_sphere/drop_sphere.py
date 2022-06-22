from math import ceil
import torch
from typing import Optional
from torch import nn

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery


class DropSphereNd(nn.Module):

    def __init__(self, dim: int, rate: float, embed_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.rate = rate
        self.embed_dim = embed_dim
        self.register_buffer('embeds', torch.randn(embed_dim, dim))

    def forward(self, x):
        if not self.training:
            return x
        n, c = x.shape[:2]
        embeds = torch.randn(n, self.embed_dim, device=x.device)
        activ = torch.einsum('ne,ed->nd', [embeds, self.embeds])
        power = 1 - self.rate
        index = ceil(c ** power)
        mins = activ.sort(1).values[:, index:index + 1]
        keep = (mins <= activ).type(x.dtype)
        shape = (n, c) + (1,) * (x.ndim - 2)
        keep = keep.view(shape)
        return x * keep * c / (c - index)


def replace(module: nn.Module, index: int):
    if not index:
        return module
    return nn.Sequential(
        DropSphereNd(module.in_channels, 0.9),
        module,
    )


def apply_drop_sphere(model: nn.Module):
    transforms = {
        nn.Conv1d: replace,
        nn.Conv2d: replace,
        nn.Conv3d: replace,
    }
    module_surgery.replace_module_classes(model, transforms)
    return model


class DropSphere(Algorithm):
    """DropSphere.

    Args:
        rate (float): Rate. Default: ``0.9``.
    """

    def __init__(self, rate: float = 0.9) -> None:
        self.rate = rate

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        assert state.model is not None, 'Model must be in state'
        apply_drop_sphere(state.model)
