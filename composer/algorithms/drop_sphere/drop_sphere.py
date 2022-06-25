from math import ceil
import numpy as np
import torch
from typing import Optional
from torch import nn

from composer.core import Algorithm, Event, State
from composer.loggers import Logger
from composer.utils import module_surgery


class DropSphereNd(nn.Module):
    """DropSphere module.

    Args:
        rate (float): Usage rate in training mode (from 0 to 1). (Default: ``0.1``)
        power (float): Channels to not drop, as a power of all channels (from 0 to 1). (Default: ``0.9``)
        embed_dim (int): Dimensionality of channel embeddings. (Default: ``16``)
    """

    def __init__(self, rate: float = 0.1, power: float = 0.9, embed_dim: int = 16):
        super().__init__()
        self.rate = rate
        self.power = power
        self.embed_dim = embed_dim
        self.was_training = False
        self.many_epochs = 100
        self.epoch = -1

    def forward(self, x):
        if not self.was_training and self.training:
            self.epoch += 1
        self.was_training = self.training

        if not self.training:
            return x

        progress = np.tanh(1.5 * self.epoch / self.many_epochs) ** 2
        if self.rate * progress < np.random.uniform():
            return x

        if not hasattr(self, 'embeds'):
            self.register_buffer('embeds', torch.randn(self.embed_dim, x.shape[1], device=x.device))

        n, c = x.shape[:2]
        embeds = torch.randn(n, self.embed_dim, device=x.device)
        activ = torch.einsum('ne,ed->nd', [embeds, self.embeds])

        index = ceil(c ** self.power) - 1
        maxes = activ.sort(1).values[:, index:index + 1]

        shape = (n, c) + (1,) * (x.ndim - 2)
        keep = (activ <= maxes).type(x.dtype).view(shape)
        return x * keep / keep.mean()


def apply_drop_sphere(model: nn.Module, rate: float = 0.1, power: float = 0.9, embed_dim: int = 16):
    """Apply drop sphere to the model.

    Args:
        rate (float): Usage rate in training mode (from 0 to 1). (Default: ``0.1``)
        power (float): Channels to not drop, as a power of all channels (from 0 to 1). (Default: ``0.9``)
        embed_dim (int): Dimensionality of channel embeddings. (Default: ``16``)
    """
    def replace(module: nn.Module, index: int):
        if not index:
            return module
        return nn.Sequential(
            DropSphereNd(rate, power, embed_dim),
            module,
        )

    transforms = {
        nn.Conv1d: replace,
        nn.Conv2d: replace,
        nn.Conv3d: replace,
    }
    module_surgery.replace_module_classes(model, transforms)
    return model


class DropSphere(Algorithm):
    """DropSphere algorithm.

    Args:
        rate (float): Usage rate in training mode (from 0 to 1). (Default: ``0.1``)
        power (float): Channels to not drop, as a power of all channels (from 0 to 1). (Default: ``0.9``)
        embed_dim (int): Dimensionality of channel embeddings. (Default: ``16``)
    """

    def __init__(self, rate: float = 0.1, power: float = 0.9, embed_dim: int = 16):
        self.rate = rate
        self.power = power
        self.embed_dim = embed_dim

    def match(self, event: Event, state: State) -> bool:
        return event == Event.INIT

    def apply(self, event: Event, state: State, logger: Optional[Logger] = None) -> None:
        apply_drop_sphere(state.model, self.rate, self.power, self.embed_dim)
