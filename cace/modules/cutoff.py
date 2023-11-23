###########################################################################################
# Radial basis cutoff
# modified from mace/mace/modules/radials.py and schnetpack/src/schnetpack/nn/cutoff.py
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import numpy as np
import torch
import torch.nn as nn

__all__ = ["CosineCutoff", "MollifierCutoff", "PolynomialCutoff", "SwitchFunction"]

def cosine_cutoff(input: torch.Tensor, cutoff: torch.Tensor):
    """ Behler-style cosine cutoff.

        .. math::
           f(r) = \begin{cases}
            0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
              & r < r_\text{cutoff} \\
            0 & r \geqslant r_\text{cutoff} \\
            \end{cases}

        Args:
            cutoff (float, optional): cutoff radius.

        """

    # Compute values of cutoff function
    input_cut = 0.5 * (torch.cos(input * np.pi / cutoff) + 1.0)
    # Remove contributions beyond the cutoff radius
    input_cut *= (input < cutoff).float()
    return input_cut


class CosineCutoff(nn.Module):
    r""" Behler-style cosine cutoff module.

    .. math::
       f(r) = \begin{cases}
        0.5 \times \left[1 + \cos\left(\frac{\pi r}{r_\text{cutoff}}\right)\right]
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    """

    def __init__(self, cutoff: float):
        """
        Args:
            cutoff (float, optional): cutoff radius.
        """
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype()))

    def forward(self, input: torch.Tensor):
        return cosine_cutoff(input, self.cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"

def mollifier_cutoff(input: torch.Tensor, cutoff: torch.Tensor, eps: torch.Tensor):
    r""" Mollifier cutoff scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}

    Args:
        cutoff: Cutoff radius.
        eps: Offset added to distances for numerical stability.

    """
    mask = (input + eps < cutoff).float()
    exponent = 1.0 - 1.0 / (1.0 - torch.pow(input * mask / cutoff, 2))
    cutoffs = torch.exp(exponent)
    cutoffs = cutoffs * mask
    return cutoffs


class MollifierCutoff(nn.Module):
    r""" Mollifier cutoff module scaled to have a value of 1 at :math:`r=0`.

    .. math::
       f(r) = \begin{cases}
        \exp\left(1 - \frac{1}{1 - \left(\frac{r}{r_\text{cutoff}}\right)^2}\right)
          & r < r_\text{cutoff} \\
        0 & r \geqslant r_\text{cutoff} \\
        \end{cases}
    """

    def __init__(self, cutoff: float, eps: float = 1.0e-7):
        """
        Args:
            cutoff: Cutoff radius.
            eps: Offset added to distances for numerical stability.
        """
        super().__init__()
        self.register_buffer("cutoff",  torch.tensor(cutoff, dtype=torch.get_default_dtype()))
        self.register_buffer("eps",  torch.tensor(eps, dtype=torch.get_default_dtype()))

    def forward(self, input: torch.Tensor):
        return mollifier_cutoff(input, self.cutoff, self.eps)

    def __repr__(self):
        return f"{self.__class__.__name__}(eps={self.eps}, cutoff={self.cutoff})"

def _switch_component(
    x: torch.Tensor, ones: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """
    Basic component of switching functions.

    Args:
        x (torch.Tensor): Switch functions.
        ones (torch.Tensor): Tensor with ones.
        zeros (torch.Tensor): Zero tensor

    Returns:
        torch.Tensor: Output tensor.
    """
    x_ = torch.where(x <= 0, ones, x)
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_))


class SwitchFunction(nn.Module):
    """
    Decays from 1 to 0 between `switch_on` and `switch_off`.
    """

    def __init__(self, switch_on: float, switch_off: float):
        """

        Args:
            switch_on (float): Onset of switch.
            switch_off (float): Value from which on switch is 0.
        """
        super(SwitchFunction, self).__init__()
        self.register_buffer("switch_on",  torch.tensor(switch_on, dtype=torch.get_default_dtype()))
        self.register_buffer("switch_off",  torch.tensor(switch_off, dtype=torch.get_default_dtype()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Args:
            x (torch.Tensor): tensor to which switching function should be applied to.

        Returns:
            torch.Tensor: switch output
        """
        x = (x - self.switch_on) / (self.switch_off - self.switch_on)

        ones = torch.ones_like(x)
        zeros = torch.zeros_like(x)
        fp = _switch_component(x, ones, zeros)
        fm = _switch_component(1 - x, ones, zeros)

        f_switch = torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm / (fp + fm)))
        return f_switch

    def __repr__(self):
        return f"{self.__class__.__name__}(switch_on={self.switch_on}, switch_off={self.switch_off})"

class PolynomialCutoff(nn.Module):
    """
    Klicpera, J.; Groß, J.; Günnemann, S. Directional Message Passing for Molecular Graphs; ICLR 2020.
    Equation (8)
    """

    p: torch.Tensor
    cutoff: torch.Tensor

    def __init__(self, cutoff: float, p=6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        envelope = (
                1.0
                - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x / self.cutoff, self.p)
                + self.p * (self.p + 2.0) * torch.pow(x / self.cutoff, self.p + 1)
                - (self.p * (self.p + 1.0) / 2) * torch.pow(x / self.cutoff, self.p + 2)
        )
        return envelope * (x < self.cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, cutoff={self.cutoff})"
