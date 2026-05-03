import torch
from typing import Optional, Dict, List, Callable, Tuple, Union

#Most of this is taken from HotPP https://doi.org/10.1038/s41467-024-51886-6

def expand_to(t     : torch.Tensor,
              n_dim : int,
              dim   : int=-1) -> torch.Tensor:
    """Expand dimension of the input tensor t at location 'dim' until the total dimention arrive 'n_dim'

    Args:
        t (torch.Tensor): Tensor to expand
        n_dim (int): target dimension
        dim (int, optional): location to insert axis. Defaults to -1.

    Returns:
        torch.Tensor: Expanded Tensor
    """
    while len(t.shape) < n_dim:
        t = torch.unsqueeze(t, dim=dim)
    return t


@torch.jit.script
def _aggregate_new(T1: torch.Tensor,
                   T2: torch.Tensor,
                   way1 : int,
                   way2 : int,
                   way3 : int,
                   ) -> torch.Tensor:
    #inputs are li, lr, lo
    coupling_way = (way1 + way2 - way3) // 2 #lc
    n_way = way1 + way2 - coupling_way + 2 #plus 2 is for E, C, so this is lo + lc (+ 2)
    output_tensor = expand_to(T1, n_way, dim=-1) * expand_to(T2, n_way, dim=2)
    # T1:  [n_edge, n_channel, n_dim, n_dim, ...,     1] 
    # T2:  [n_edge, n_channel,     1,     1, ..., n_dim]  
    # with (way1 + way2 - coupling_way) dim after n_channel
    # We should sum up (coupling_way) n_dim
    if coupling_way > 0:
        sum_axis = [i for i in range(way1 - coupling_way + 2, way1 + 2)]
        output_tensor = torch.sum(output_tensor, dim=sum_axis)
    return output_tensor


def single_tensor_product(x : torch.Tensor,
                          y : torch.Tensor, #lr
                          combination : Tuple[int], #(li,lr,lout)
                         ) -> torch.Tensor:
        x_way, y_way, z_way = combination
        return _aggregate_new(x, y, x_way, y_way, z_way)
