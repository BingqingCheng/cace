import torch
import torch.nn as nn
import lightning as L
from typing import Optional, Dict, List, Callable, Tuple, Union

from .tensornet_utils import multi_outer_product
from .tensornet_utils import _aggregate_new, expand_to, find_distances, find_moment, _scatter_add, single_tensor_product, layer_norm, normalize_tensors

class TensorLinearMixing(nn.Module):
    def __init__(self,
                 n_out : int,
                 lomax : int,
                 ) -> None:
        super().__init__()
        self.linear_list = nn.ModuleList([
            nn.LazyLinear(n_out, bias=False) for l in range(lomax + 1)
        ])

    def forward(self,
                input_tensors : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for l, linear in enumerate(self.linear_list):
            input_tensor = torch.transpose(input_tensors[l], 1, -1)
            output_tensor = linear(input_tensor)
            output_tensors[l] = torch.transpose(output_tensor, 1, -1)
        return output_tensors

class TensorProductLayer(nn.Module):
    def __init__(self, nc,
                 max_x_way      : int=2,
                 max_y_way      : int=2,
                 max_z_way      : int=2,
                 zstack         : bool=False,
                 stacking       : bool=False,
                 ) -> None:
        #lin, lr, lout
        super().__init__()
        self.stacking = stacking
        self.zstack = zstack
        self.combinations = []
        for x_way in range(max_x_way + 1):
            for y_way in range(max_y_way + 1):
                for z_way in range(abs(y_way - x_way), min(max_z_way, x_way + y_way) + 1, 2):
                    self.combinations.append((x_way, y_way, z_way))

    def forward(self,
                x : Dict[int, torch.Tensor],
                y : Dict[int, torch.Tensor],
                ) -> Dict[int, torch.Tensor]:
        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        for x_way, y_way, z_way in self.combinations:
            if x_way not in x or y_way not in y:
                continue
            output_tensor = _aggregate_new(x[x_way], y[y_way], x_way, y_way, z_way)
            if z_way not in output_tensors:
                output_tensors[z_way] = output_tensor
            else:
                if self.stacking:
                    output_tensors[z_way] = torch.hstack([output_tensors[z_way],output_tensor])
                elif self.zstack and (z_way == 0):
                    output_tensors[z_way] = torch.hstack([output_tensors[z_way],output_tensor])
                else:
                    output_tensors[z_way] += output_tensor
        return output_tensors

class TensorActivationGate(nn.Module):
    def __init__(self,l_out_list:List[int]) -> None:
        super().__init__()
        self.lomax = len(l_out_list) - 1
        self.net0 = nn.Sequential(nn.LazyLinear(l_out_list[0],bias=True),nn.SiLU())
        self.norm_net_list = nn.ModuleList([
            nn.Sequential(nn.LazyLinear(nc,bias=True),nn.Sigmoid()) for nc in l_out_list[1:] 
        ])
    
    def forward(self,input_tensors : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        #Make mlp feed
        mlp_feed = [input_tensors[0]]
        for l in range(1,self.lomax+1):
            input_tensor_ = input_tensors[l].reshape(input_tensors[l].shape[0], input_tensors[l].shape[1], -1)
            norm = torch.sum(input_tensor_ ** 2, dim=2)
            mlp_feed.append(norm)
        mlp_feed = torch.hstack(mlp_feed)

        output_tensors = torch.jit.annotate(Dict[int, torch.Tensor], {})
        output_tensors[0] = self.net0(mlp_feed)
        for l in range(1,self.lomax+1):
            mlp_out = self.norm_net_list[l-1](mlp_feed)
            output_tensors[l] = input_tensors[l] * expand_to(mlp_out,l+2)
        return output_tensors

class TensorFeedForward(nn.Module):
    def __init__(self,nc,lomax) -> None:
        super().__init__()
        self.lomax = lomax
        self.mix1 = TensorLinearMixing(nc,lomax)
        self.gate = TensorActivationGate([nc]*(lomax+1))
        self.mix2 = TensorLinearMixing(nc,lomax)

    def forward(self,input_tensors : Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        output_tensors = self.mix1(input_tensors)
        output_tensors = self.gate(output_tensors)
        output_tensors = self.mix2(output_tensors)
        return output_tensors
