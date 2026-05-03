from typing import Dict
import torch
from torch import nn

from .tensornet import TensorFeedForward

__all__ = ['TensorReadout']

class TensorReadout(nn.Module):
    """
    Predicts dipoles/quadrupoles directly from node_feats_l
    """

    def __init__(
        self,
        max_l: int,
        feature_key: str = 'node_feats_l',
        l0_key: 'str | list[str]' = 'scalar',
        l1_key: 'str | list[str]' = 'vector',
        l2_key: 'str | list[str]' = 'quadrupole',
        n_channel: int = 1,
        l0_output_scale: float = 1.0,
        l1_output_scale: float = 1.0,
        l2_output_scale: float = 1.0,
    ):
        """
        Args:        
        """
        super().__init__()
        assert max_l >= 1, "max_l must be at least 1 for vector prediction."
        assert max_l <= 2, "max_l greater than 2 is not supported for direct prediction."
        self.max_l = max_l
        self.feature_key = feature_key
        self.l0_key = [l0_key] if isinstance(l0_key, str) else l0_key
        self.l1_key = [l1_key] if isinstance(l1_key, str) else l1_key
        self.l2_key = [l2_key] if isinstance(l2_key, str) else l2_key
        self.l0_output_scale = l0_output_scale
        self.l1_output_scale = l1_output_scale
        self.l2_output_scale = l2_output_scale

        self.model_outputs = []
        for key in self.l0_key:
            self.model_outputs.append(key)
        for key in self.l1_key:
            self.model_outputs.append(key)
        if max_l >= 2:
            for key in self.l2_key:
                self.model_outputs.append(key)       

        self.required_derivatives = []
        nout = max(len(self.l0_key),len(self.l1_key),len(self.l2_key))
        self.n_channel = n_channel
        tff_dim = self.n_channel * nout
        self.tensor_feed_forward = TensorFeedForward(tff_dim, lomax=max_l)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'n_channel'):
            self.n_channel = 1
        if isinstance(self.l0_key, str):
            self.l0_key = [self.l0_key]
        if isinstance(self.l1_key, str):
            self.l1_key = [self.l1_key]
        if isinstance(self.l2_key, str):
            self.l2_key = [self.l2_key]
 
        if self.feature_key not in data:
            raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
        features = data[self.feature_key] #{0: l=0, 1:l=1, 2:l=2...}

        assert 1 in features, f"Features for l=1 not found in data dictionary under key {self.feature_key}."
        if self.max_l >= 2:
            assert 2 in features, f"Features for l=2 not found in data dictionary under key {self.feature_key}."

        out = self.tensor_feed_forward(features)
        start, stop = 0, self.n_channel
        for k in self.l0_key:
            data[k] = (out[0][:, start:stop] * self.l0_output_scale).squeeze()
            start += self.n_channel
            stop += self.n_channel
        start, stop = 0, self.n_channel
        for k in self.l1_key:
            data[k] = (out[1][:, start:stop] * self.l1_output_scale).squeeze()
            start += self.n_channel
            stop += self.n_channel
        if self.max_l >= 2:
            start, stop = 0, self.n_channel
            for k in self.l2_key:
                data[k] = (out[2][:, start:stop] * self.l2_output_scale).squeeze()
                start += self.n_channel
                stop += self.n_channel
           
       
        return data 

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            )
