import numpy as np
import torch

def elementwise_multiply_2tensors(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiplication of two 2D tensors
    :param a: (N, A) tensor
    :param b: (N, B) tensor
    :return: (N, A, B) tensor
    """
    # expand the dimenstions for broadcasting
    a_expanded = a.unsqueeze(2)
    b_expanded = b.unsqueeze(1)
    # multiply
    return a_expanded * b_expanded

def elementwise_multiply_3tensors(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Elementwise multiplication of three 2D tensors
    :param a: (N, A) tensor
    :param b: (N, B) tensor
    :param c: (N, C) tensor
    :return: (N, A, B, C) tensor
    """
    # expand the dimenstions for broadcasting
    a_expanded = a.unsqueeze(2).unsqueeze(3)
    b_expanded = b.unsqueeze(1).unsqueeze(3)
    c_expanded = c.unsqueeze(1).unsqueeze(2)
    # multiply
    return a_expanded * b_expanded *  c_expanded
    
def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.cpu().detach().numpy()

def voigt_to_matrix(t: torch.Tensor):
    """
    Convert voigt notation to matrix notation
    :param t: (6,) tensor or (3, 3) tensor
    :return: (3, 3) tensor
    """
    if t.shape == (3, 3):
        return t

    return torch.tensor(
        [[t[0], t[5], t[4]], [t[5], t[1], t[3]], [t[4], t[3], t[2]]], dtype=t.dtype
    )

def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding with <num_classes> classes from <indices>
    :param indices: (N x 1) tensor
    :param num_classes: number of classes
    :param device: torch device
    :return: (N x num_classes) tensor
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)
