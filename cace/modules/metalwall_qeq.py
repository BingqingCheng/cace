from typing import Dict, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ewald import EwaldPotential

__all__ = ['MetalWallQEQ']

class MetalWallQEQ(nn.Module):
    def __init__(self,
                 metal_atomic_numbers: Union[int, List[int]],  # atomic number of the metal atoms
                 dl=2.0,  # grid resolution
                 sigma=1.0, #1/1.805132, #1.0,  # width of the Gaussian on each atom
                 external_field = None, # external field
                 external_field_direction: int = 2, # external field direction, 0 for x, 1 for y, 2 for z
                 external_field_norm_factor: float = (1./90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
                 external_field_on: bool = True, # whether to directly apply efield on non-metal atoms
                 external_field_potential_on: bool = True, # whether to explicitly enforce the potential drop of the electrode
                 feature_key: str = 'q',
                 chi_key: str = None, # if None than all zero
                 output_key: str = 'q_mw',
                 system_charge: Union[float, str] = 0.0,  # Key for system charge in data
                 system_charge_norm_factor: float = (90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
                 scaling_factor: float = 1.0,  # set to be \sqrt{\epsilon_r} of the electrolyte. All charges in the electrolyte are scaled by Q^les = q/scaling_factor
                 J_processing: str = 'square', # 'square' or 'softplus'
                 J_init: float = 0.0, # initial value for J
                 use_cache: bool = True
                 ):
        super().__init__()
        self.ep = EwaldPotential(dl=dl,
                    sigma=sigma,
                    exponent=1, # coulumb
                    feature_key=feature_key,
                    aggregation_mode='sum',
                    remove_self_interaction=False,
                    output_key=None,
                    compute_field=True)
        self.ep.model_outputs = []
        if isinstance(metal_atomic_numbers, int):
            metal_atomic_numbers = [metal_atomic_numbers]
        self.metal_atomic_numbers = metal_atomic_numbers
        Z_index_map = torch.full((max(metal_atomic_numbers) + 1,), -1)
        for i, z in enumerate(metal_atomic_numbers):
            Z_index_map[z] = i
        self.register_buffer('Z_index_map', Z_index_map)

        init_J = J_init * torch.ones(len(metal_atomic_numbers)) # initialize for all elements
        self.J_raw = nn.Parameter(data=init_J, requires_grad=True)
        self.J_processing = J_processing

        self.AJl = None
        self.r = None
        self._J_last = None
        self.use_cache = use_cache
        
        self.feature_key = feature_key
        self.chi_key = chi_key
        self.output_key = output_key
        self.system_charge = system_charge
        self.system_charge_norm_factor = system_charge_norm_factor

        self.model_outputs = [output_key]
        self.scaling_factor = scaling_factor

        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.external_field_norm_factor = external_field_norm_factor
        self.external_field_on = external_field_on
        self.external_field_potential_on = external_field_potential_on
       
    def forward(self, data: Dict[str, torch.Tensor], **kwargs):

        if not hasattr(self, 'AJl'):
            self.AJl = None
        if not hasattr(self, 'use_cache'):
            self.use_cache = True
        if not hasattr(self, 'J_processing'):
            self.J_processing = 'square'
        if not hasattr(self, 'system_charge_norm_factor'):
            self.system_charge_norm_factor = (90.0474)**0.5

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]
            
        box_all = data['cell'].view(-1, 3, 3)
        r_all = data['positions']
        q_all = data[self.feature_key]
        chi_all = data[self.chi_key] if self.chi_key is not None else torch.zeros_like(q_all)
        atomic_numbers_all = data['atomic_numbers']
        if self.J_processing == 'square':
            J_elem = torch.square(self.J_raw) # positive
        elif self.J_processing == 'softplus':
            J_elem = F.softplus(self.J_raw)

        idx = self.Z_index_map[atomic_numbers_all]
        J_i = J_elem[idx]

        if q_all.dim() == 1:
            q_all = q_all.unsqueeze(1)
        if chi_all.dim() == 1:
            chi_all = chi_all.unsqueeze(1)

        # Check the input dimension
        n, d = r_all.shape
        assert d == 3, 'r dimension error'
        assert n == q_all.size(0), 'q dimension error'
        

        # set the charges to all metal elements to zero
        metal_z = torch.tensor(self.metal_atomic_numbers,
                       device=atomic_numbers_all.device,
                       dtype=atomic_numbers_all.dtype)
        all_metal_index = torch.isin(atomic_numbers_all, metal_z)  # shape [n], bool
        q_all[all_metal_index] = 0.0

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        if isinstance(self.system_charge, (int, float)):
            system_charge = torch.full((len(unique_batches),), 
                                       self.system_charge * self.system_charge_norm_factor, 
                                       device=data['positions'].device)
        elif self.system_charge in data:
            system_charge = data[self.system_charge] * self.system_charge_norm_factor
        else:
            raise ValueError(f'system_charge {self.system_charge} not found in data')

                            
        results = []
        energy_external_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_now, q_now, atomic_numbers, cell = r_all[mask], q_all[mask], atomic_numbers_all[mask], box_all[i]
            system_charge_now = system_charge[i]

            metal_index =  torch.isin(atomic_numbers, metal_z) # metal atoms in this configuration
            J_i_now = J_i[mask][metal_index]
            chi_now = chi_all[mask][metal_index]
            electrolyte_index = ~metal_index # non-metal atoms in this configuration


            energy_external = torch.zeros(1, device=q_now.device)

            if metal_index.sum() == 0:
                q_combined = q_now.clone()
                # If there are no metal atoms, we just return the original charges
                results.append(q_combined)
            else:
                # get the A matrix
                r = r_now[metal_index, :]
                B_ext = torch.zeros_like(r[:, self.external_field_direction])
                # if AJl cache is stale -> recompute
                need_recompute = (
                    self.AJl is None
                    or self.r is None
                    or self.r.shape != r.shape
                    or not torch.allclose(self.r, r)
                    or self._J_last is None
                    or not torch.allclose(self._J_last, J_i_now)
                    or not self.use_cache
                )

                # if the positions of metal atoms haven't changed, we use the stored S matrix
                if need_recompute:
                    if self.use_cache:
                        self.AJl = self._compute_S_matrix(r.detach(), cell.detach(), J_i_now.detach())
                    else:
                        self.AJl = self._compute_S_matrix(r.detach(), cell.detach(), J_i_now)
                    self.r = r.detach()
                    self._J_last = J_i_now.detach()

                if electrolyte_index.sum() > 0:
                    _, f_now = self.ep.compute_potential_triclinic(r_now, 
                                                           q_now, 
                                                           cell, 
                                                           compute_field=True)
                else:
                    f_now = torch.zeros_like(q_now)

                if self.external_field is not None:
                    # assumes that the metal electrodes are on the sides, and the electrolytes are in the middle
                    # E = \dPhi / L
                    # energy_external = - E * r * q
                    if self.external_field_on:
                        energy_external = - self.external_field * torch.sum(r_now[electrolyte_index, self.external_field_direction].unsqueeze(1) * q_now[electrolyte_index]).unsqueeze(0) * self.external_field_norm_factor
                    if self.external_field_potential_on:
                        lz = cell[self.external_field_direction, self.external_field_direction]
                        r_wrap = r[:, self.external_field_direction] / lz
                        r_wrap = r_wrap - torch.round(r_wrap)
                        r_wrap = r_wrap * lz
                        B_ext = self.external_field * r_wrap * self.external_field_norm_factor
                # the external field generated by electrolyte atoms needs to be scaled by \sqrt{\epsilon_r}
                B_mat = - f_now[metal_index, :] * self.scaling_factor  + B_ext.unsqueeze(1)
                chi_vector = torch.cat([B_mat - chi_now, torch.tensor([system_charge_now], device=q_now.device, dtype=q_now.dtype).unsqueeze(1)])

                q_mw = q_now.clone()
                q_sol_lambda = self.AJl @ chi_vector
                # we then scale the true qs to get q^les
                q_mw[metal_index] = q_sol_lambda[:-1] / self.scaling_factor
                results.append(q_mw)

            energy_external_results.append(energy_external)

        data['mw_energy_external'] = torch.cat(energy_external_results, dim=0)
        data[self.output_key] = torch.cat(results, dim=0)
        
        return data
        
    def _compute_S_matrix(self, r, cell, J):
        N = r.shape[0]
        dtype = r.dtype
        device = r.device

        q_eye = torch.eye(N, dtype=dtype, device=device)
        _, A_mat = self.ep.compute_potential_triclinic(r, q_eye, cell, compute_field=True)

        top_left = A_mat.to(J.dtype) + torch.diag(J)  # depends on J
        top_right = torch.ones((N, 1), dtype=top_left.dtype, device=device)
        bot_left  = torch.ones((1, N), dtype=top_left.dtype, device=device)
        bot_right = torch.zeros((1, 1), dtype=top_left.dtype, device=device)

        coeffs = torch.cat(
            [torch.cat([top_left, top_right], dim=1),
             torch.cat([bot_left, bot_right], dim=1)],
            dim=0
        )

        I = torch.eye(N + 1, dtype=coeffs.dtype, device=device)
        S = torch.linalg.solve(coeffs, I)
        return S

    def set_cache(self, use_cache: bool):
        self.use_cache = use_cache
