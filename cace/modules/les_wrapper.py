import torch
import torch.nn as nn
from typing import Dict, Sequence, Union

__all__ = ['LesWrapper']

class LesWrapper(nn.Module):
    """
    A wrapper for the LES library that does long-range interactions and BECs
    Note that CACE has its own internal implementation of the LES algorithm
    so it is not necessary to use this wrapper in CACE.
    """
    def __init__(self,
                 feature_key: Union[str, Sequence[int]] = 'node_feats',
                 energy_key: str = 'LES_energy',
                 charge_key: str = 'LES_charge',
                 bec_key: str = 'LES_BEC',
                 compute_energy: bool = True,
                 compute_bec: bool = False,
                 bec_output_index: int = None, # option to compute BEC along one axis
                 ):
        super().__init__()
        from les import Les
        self.les = Les(les_arguments={})
 
        self.feature_key = feature_key
        self.energy_key = energy_key
        self.charge_key = charge_key
        self.bec_key = bec_key
        self.bec_output_index = bec_output_index

        self.compute_energy = compute_energy        
        self.compute_bec = compute_bec
        self.model_outputs = [charge_key]
        if compute_energy:
            self.model_outputs.append(energy_key)
        if compute_bec:
            self.model_outputs.append(bec_key)

    def set_compute_energy(self, compute_energy: bool):
        self.compute_energy = compute_energy

    def set_compute_bec(self, compute_bec: bool):
        self.compute_bec = compute_bec

    def set_bec_output_index(self, bec_output_index: int):
        self.bec_output_index = bec_output_index

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:

        # reshape the feature vectors
        if isinstance(self.feature_key, str):
            if self.feature_key not in data:
                raise ValueError(f"Feature key {self.feature_key} not found in data dictionary.")
            features = data[self.feature_key]
            features = features.reshape(features.shape[0], -1)
        elif isinstance(self.feature_key, list):
            features = torch.cat([data[key].reshape(data[key].shape[0], -1) for key in self.feature_key], dim=-1)

        result = self.les(desc=features,
            positions=data['positions'],
            cell=data['cell'].view(-1, 3, 3),
            batch=data["batch"],
            compute_energy=self.compute_energy,
            compute_bec=self.compute_bec,
            bec_output_index=self.bec_output_index,
            )

        data[self.charge_key] = result['latent_charges']
        if self.compute_energy:
            data[self.energy_key] = result['E_lr']
        if self.compute_bec:
            data[self.bec_key] = result['BEC']
        return data
