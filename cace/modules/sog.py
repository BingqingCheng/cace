"""SOG long-range potential module."""

import torch
import torch.nn as nn
from typing import Dict

__all__ = ["SOGPotential"]


class SOGPotential(nn.Module):
    def __init__(self,
                 N_dl=1,  # Fourier modes
                 bandwidth_num = 12,
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'SOG_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 Periodic: bool = False,
                 ):
        super().__init__()
        self.N_dl = N_dl
        self.bandwidth_num = bandwidth_num

        self.amplitude_1 = torch.tensor(
            [0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001],
            dtype=torch.float32,
        )
        self.shift_1 = torch.tensor(
            [2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9],
            dtype=torch.float32,
        )

        self.norm_factor = torch.tensor(1.0)

        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')
        self.charge_neutral_lambda = charge_neutral_lambda

        self.twopi = 2.0 * torch.pi
        self.Periodic = Periodic

        if self.Periodic:
            self.wl = torch.nn.Parameter(
                self.amplitude_1 * (torch.sqrt(torch.tensor(torch.pi)) ** 3) * self.shift_1**3
            )
            self.sl = torch.nn.Parameter(-torch.log(2 / self.shift_1))
        else:
            self.wl = torch.nn.Parameter(self.amplitude_1)
            self.sl = torch.nn.Parameter(self.shift_1)

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        #print("cell:",box)

        r = data['positions'] # (total_atom_number_of_all_configurations_in_batch, 3)
        #print(r.shape)

        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print(q.shape) # (total_atom_number_of_all_configurations_in_batch, number_of_q_layers)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices. Batch_now saves the corresponding configuration index [0 0 ... 0 1 ... 1 2 ... 2]. Unique is used to get the total number of configurations in the batch
        #print(batch_now)
        #print(batch_now.shape, unique_batches.shape)
        results = []
        field_results = []
        for i in unique_batches:
            mask = (batch_now == i)  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i] # Extract the atomic information for each configuration.
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)

            if self.Periodic:
                # self.wl = (torch.nn.Parameter(self.amplitude_1*(torch.sqrt(torch.tensor(torch.pi))**3)*self.shift_1**3)
                # self.sl = (torch.nn.Parameter(-torch.log(2/self.shift_1))
                pot = self.compute_potential_SOG_triclinic(r_raw_now, q_now, box_now)
            else:
                # self.wl = torch.nn.Parameter(self.amplitude_1
                # self.sl = torch.nn.Parameter(self.shift_1
                pot = self.compute_potential_SOG_realspace(r_raw_now, q_now)

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0

            results.append(pot + pot_neutral)

        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            # field_results.append(field)
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data
 

    def compute_potential_SOG_triclinic(self, r_raw, q, box):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device
        
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T
        
        
        norms = torch.norm(box, dim=1)
        Nk = [max(1, int(n.item() / self.N_dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = (nvec.float() @ G).to(device)
        
        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]


        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        if q.dim() == 1:  
            q = q.unsqueeze(1)
         #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]
        # print("S_k_sq shape:",S_k_sq.shape)
        min_term = -1 / torch.exp(-2 * self.sl)  # 计算指数衰减
        min_term = min_term.view(1, 1, 1, -1)  # 扩展维度
        kfac = self.wl.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)  # 计算 SOG 频谱响应
        # print("multiplier:",multiplier.shape,multiplier)c
        kfac = kfac.sum(dim=-1)  # 在 SOG 频谱维度求和


        volume = torch.det(box)
        pot = (factors * kfac * S_k_sq).sum(dim=1) / volume
        pot = pot.view(-1)
        # print("pot shape:",pot.shape)
        if self.remove_self_interaction:
            diag_sum = kfac.sum(dim=-1).sum(dim=-1).sum(dim=-1) / (2 * volume)
            pot -= torch.sum(q**2)*diag_sum

        return pot * self.norm_factor


    def compute_potential_SOG_realspace(self, r_raw, q):

        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        epsilon = 1e-6
        torch.diagonal(r_ij).add_(epsilon)


        min_term = -1/self.sl**2
        min_term = min_term.view(1, 1, -1)  # 扩展维度
        convergence_func_ij = self.wl.view(1, 1, -1) * torch.exp( torch.square(r_ij_norm).unsqueeze(2) * min_term)
        convergence_func_ij = torch.sum(convergence_func_ij, dim=2)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)

        idx = torch.arange(convergence_func_ij.size(0))
        convergence_func_ij[idx, idx] = 0

        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0        

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False :
            pot += torch.sum(q ** 2) * self.wl.sum() / self.twopi / 2.0 

        return pot* self.norm_factor


    def add_external_field(self, r_raw, q, box, direction_index, external_field):
        external_field_norm_factor = (self.norm_factor/90.0474)**0.5
        # wrap in box
        r = r_raw[:, direction_index] / box[direction_index]
        r =  r - torch.round(r)
        r = r * box[direction_index]
        return external_field * torch.sum(q * r.unsqueeze(1)) * external_field_norm_factor

    def change_external_field(self, external_field):
        self.external_field = external_field

    def is_orthorhombic(self, cell_matrix):
        diag_matrix = torch.diag(torch.diagonal(cell_matrix))
        is_orthorhombic = torch.allclose(cell_matrix, diag_matrix, atol=1e-6)
        return is_orthorhombic
