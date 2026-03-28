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


def multi_outer_product(v: torch.Tensor,
                        n: int) -> torch.Tensor:
    """Calculate 'n' times outer product of vector 'v'

    Args:
        v (torch.TensorType): vector or vectors ([n_dim] or [..., n_dim])
        n (int): outer prodcut times, will return [...] 1 if n = 0

    Returns:
        torch.Tensor: OvO
    """
    out = torch.ones_like(v[..., 0]) #very slick, cool!
    for _ in range(n):
        out = out[..., None] * expand_to(v, len(out.shape) + 1, dim=len(v.shape) - 1)
    return out


def find_distances(data  : Dict[str, torch.Tensor],) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx_i = data["edge_index"][0]
    idx_j = data["edge_index"][1]
    if 'rij' not in data:
        data['rij'] = data['positions'][idx_j] - data['positions'][idx_i]
    if 'dij' not in data:
        data['dij'] = torch.norm(data['rij'], dim=-1)
    if 'uij' not in data:
        data['uij'] = data['rij'] / data['dij'].unsqueeze(-1)
    return data['rij'], data['dij'], data['uij']


def find_moment(batch_data  : Dict[str, torch.Tensor],
                n_way       : int
                ) -> torch.Tensor:
    if 'moment' + str(n_way) not in batch_data:
        find_distances(batch_data)
        batch_data['moment' + str(n_way)] = multi_outer_product(batch_data['uij'], n_way)
    return batch_data['moment' + str(n_way)]


@torch.jit.script
def _scatter_add(x        : torch.Tensor, 
                 idx_i    : torch.Tensor, 
                 dim_size : Optional[int]=None, 
                 dim      : int = 0
                 ) -> torch.Tensor:
    shape = list(x.shape)
    if dim_size is None:
        dim_size = idx_i.max() + 1
    shape[dim] = dim_size
    tmp = torch.zeros(shape, dtype=x.dtype, device=x.device)
    y = tmp.index_add(dim, idx_i, x)
    return y


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


@torch.jit.script
def normalize_tensors(input_tensors : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l in input_tensors.keys():
            input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
            factor = 1/(torch.sum(input_tensor_ ** 2, dim=2) + 1)
            output_tensors[l] = expand_to(factor,l+2) * input_tensors[l]
        return output_tensors

@torch.jit.script
def layer_norm(input_tensors : Dict[int, torch.Tensor],eps:float=1e-10) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        input_tensors[0] = input_tensors[0] - input_tensors[0].mean(dim=-1)[:,None]
        for l in input_tensors.keys():
            input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
            rms = torch.sqrt(torch.sum(input_tensor_ ** 2,dim=-1).mean(dim=-1) + eps)
            factor = (1/rms)
            output_tensors[l] = expand_to(factor,l+2) * input_tensors[l]
        return output_tensors

