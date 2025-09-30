from typing import Dict, List
import torch
import torch.nn as nn
from .ewald import EwaldPotential

__all__ = ['ChargeEq']

class ChargeEq(nn.Module):
    def __init__(self,
                 dl: float = 1.5,
                 sigma: float = 1.0,
                 elements: List[int] = None,
                 feature_key: str = 'chi',
                 output_key: str = 'q_eq',
                 ewald_key: str = 'ewald_potential',
                 system_charge: float = 0.0,
                 remove_self_interaction: bool = True,
                 aggregation_mode: str = 'sum',
                 compute_field: bool = True,
                 norm_factor: float = (1./90.0474)**0.5, 
                 scaling_factor: float = 1.0,
                 system_charge_key: str = 'system_charge',  # Key for system charge in data
                 # the standard normal factor in accordance with the cace convention used in ewald.py
                 ):
        super().__init__()

        self.feature_key = feature_key
        self.output_key = output_key
        self.ewald_key = ewald_key
        self.model_outputs = [output_key, ewald_key]
        self.normalization_factor = norm_factor  # 1/2\epsilon_0
        self.scaling_factor = scaling_factor
        self.compute_field = compute_field
        self.system_charge = system_charge
        self.aggregation_mode = aggregation_mode
        self.system_charge_key = system_charge_key

        self.ep = EwaldPotential(
            dl=dl,
            sigma=sigma,
            remove_self_interaction=remove_self_interaction,
            aggregation_mode=aggregation_mode,
        )
        self.elements = elements
        Z_max = max(elements)
        Z_index_map = torch.full((Z_max + 1,), -1)
        for i, z in enumerate(elements):
            Z_index_map[z] = i
        self.register_buffer('Z_index_map', Z_index_map)

        init_J = torch.ones(len(elements)) # initialize J to 1 for all elements
        self.J_raw = nn.Parameter(data=init_J, requires_grad=True)


    def forward(self, data: Dict[str, torch.Tensor], **kwargs):

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        box = data['cell']
        r = data['positions']
        chi = data[self.feature_key]
        Z = data['atomic_numbers']
        element_types = torch.unique(Z)
        assert len(element_types) == len(self.elements), \
            f"Number of unique elements {len(element_types)} != expected number {len(self.elements)}."
        if chi.dim() == 1:
            chi = chi.unsqueeze(1)

        J_raw = self.J_raw
        J_elem = torch.square(J_raw) # positive
        idx = self.Z_index_map[Z]
        J_i = J_elem[idx]

        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == chi.size(0), 'chi dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        if (self.system_charge_key not in data or data[self.system_charge_key] is None
            ) and self.system_charge is not None:
            system_charge = self.system_charge
            system_charge = torch.full((len(unique_batches),), 
                                       system_charge, device=data['positions'].device)
            # print('system_charge from class:', system_charge)
        else:
            # print('system_charge from data:', data[self.system_charge_key])
            system_charge = data[self.system_charge_key]


        results = []
        ewald_results = []

        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            r_now, chi_now, box_now = r[mask], chi[mask], box[i]
            system_charge_now = system_charge[i]
            # print(f"Processing batch {i}, system_charge: {system_charge_now}")
            J_i_now = J_i[mask]
            A_now = self._compute_A_matrix(r_now, box_now)
            q_eq, lambda_eq = self._compute_q_eq(A_now, chi_now, J_i_now, system_charge_now)
            results.append(q_eq)
            ewald_energy = 0.5 * q_eq.unsqueeze(1).T @ A_now @ q_eq.unsqueeze(1)
            ewald_results.append(ewald_energy)
            # print('ewald_energy:', ewald_energy)
            # print(q_eq[:3])
            # print(q_eq.sum(axis=0), lambda_eq)
        # print(J_i_now.shape)
        # print(q_eq.shape, lambda_eq.shape)
        # print(J_i_now[:2])
        # print(A_now.shape)
        # print(A_now[:2])
        # print(A_plus_J.shape)
        # print(A_plus_J[:2])
        # print(Z[mask].shape)
        # print(Z[mask][:2])
        all_q_eq = torch.cat(results, dim=0)
        # print('all_q_eq.shape:', all_q_eq.shape)
        if all_q_eq.dim() == 1:
            all_q_eq = all_q_eq.unsqueeze(1)
        data[self.output_key] = all_q_eq
        all_ewald = torch.stack(ewald_results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(ewald_results, dim=0)
        if all_ewald.dim() != 1:
            all_ewald = all_ewald.squeeze(-1)
        data[self.ewald_key] = all_ewald
        # print('data[self.ewald_key].shape:', data[self.ewald_key].shape)
        # print('data[self.ewald_key]:', data[self.ewald_key])
        # print('J_raw', J_raw)

        return data



    def _compute_A_matrix(self, r, cell):
        N_atoms = len(r)
        q_eye = torch.eye(N_atoms, device=r.device)
        _, A_mat = self.ep.compute_potential_triclinic(r, q_eye, cell, compute_field=self.compute_field)

        return A_mat

    def _compute_q_eq(self, A_mat, chi, J, system_Q):
        device, dtype = A_mat.device, A_mat.dtype
        N_atoms = A_mat.size(0)
        A_plus_J = A_mat + torch.diag(J.to(dtype))
        # print('A_plus_J.shape:', A_plus_J.shape)
        coeffs = torch.ones((N_atoms+1, N_atoms+1), device=device, dtype=dtype)
        coeffs[:N_atoms, :N_atoms] = A_plus_J
        coeffs[N_atoms,  N_atoms]  = 0.0
        # print('coeffs.shape:', coeffs.shape)
        Q_tot = system_Q / self.normalization_factor # normalized to be consistent with ewald.py
        chi_vector = torch.cat([-chi.view(-1),
                                torch.tensor([Q_tot], device=device, dtype=dtype)])
        # print('chi_vector.shape:', chi_vector.shape)
        chi_vector = chi.unsqueeze(1) if chi.dim() == 1 else chi_vector
        # print('chi_vector.shape:', chi_vector.shape)
        sol = torch.linalg.solve(coeffs, chi_vector)
        # print('sol.shape:', sol.shape)
        q_eq = sol[:N_atoms]
        lambda_eq = sol[N_atoms]
        return q_eq, lambda_eq