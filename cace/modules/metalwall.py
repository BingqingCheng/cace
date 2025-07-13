from typing import Dict, List
import torch
import torch.nn as nn
from .ewald import EwaldPotential

__all__ = ['MetalWall']

class MetalWall(nn.Module):
    def __init__(self,
                 metal_atomic_numbers: int,
                 dl=1.,  # grid resolution
                 sigma=1/1.805132,  # width of the Gaussian on each atom
                 external_field = None, # external field
                 external_field_direction: int = 2, # external field direction, 0 for x, 1 for y, 2 for z
                 external_field_norm_factor: float = (1./90.0474)**0.5, # the standard normal factor in accordance with the cace convention used in ewald.py
                 external_field_on: bool = True, # whether to directly apply efield on non-metal atoms
                 external_field_potential_on: bool = True, # whether to explicitly enforce the potential drop of the electrode
                 feature_key: str = 'q',
                 output_key: str = 'q_mw',
                 adjust_neutrality: bool = False,
                 scaling_factor: float = 1.0  # set to be \sqrt{\epsilon_r} of the electrolyte. All charges in the electrolyte are scaled by Q^les = q/scaling_factor
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
        self.metal_atomic_numbers = metal_atomic_numbers
        self.S = None
        self.r = None
        self.A_mat = None
        
        self.feature_key = feature_key
        self.output_key = output_key
        self.adjust_neutrality = adjust_neutrality        
        self.model_outputs = [output_key]
        self.scaling_factor = scaling_factor

        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.external_field_norm_factor = external_field_norm_factor
        self.external_field_on = external_field_on
        self.external_field_potential_on = external_field_potential_on
       
    def _backward_compatibility(self): 
        if not hasattr(self, 'adjust_neutrality'):
            self.adjust_neutrality = False
        if not hasattr(self, 'scaling_factor'):
            self.scaling_factor = 1.0
        if not hasattr(self, 'A_mat'):
            self.A_mat = None
        if not hasattr(self, 'external_field'):
            self.external_field = None
        if not hasattr(self, 'external_field_direction'):
            self.external_field_direction = 2
        if not hasattr(self, 'external_field_norm_factor'):
            self.external_field_norm_factor = (1./90.0474)**0.5
        if not hasattr(self, 'external_field_on'):
            self.external_field_on = True
        if not hasattr(self, 'external_field_potential_on'):
            self.external_field_potential_on = True

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):

        self._backward_compatibility()        

        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]
            
        box_all = data['cell'].view(-1, 3, 3)
        r_all = data['positions']
        q_all = data[self.feature_key]
        atomic_numbers_all = data['atomic_numbers']
        if q_all.dim() == 1:
            q_all = q_all.unsqueeze(1)

        # Check the input dimension
        n, d = r_all.shape
        assert d == 3, 'r dimension error'
        assert n == q_all.size(0), 'q dimension error'
        

        # set the charges to all metal elements to zero
        all_metal_index = (atomic_numbers_all == self.metal_atomic_numbers)
        q_all[all_metal_index] = 0.0

        unique_batches = torch.unique(batch_now)  # Get unique batch indices
                            
        results = []
        energy_corr_results = []
        energy_external_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_now, q_now, atomic_numbers, cell = r_all[mask], q_all[mask], atomic_numbers_all[mask], box_all[i]
            
            metal_index = (atomic_numbers == self.metal_atomic_numbers)
            electrolyte_index = ~metal_index

            energy_external = torch.zeros(1, device=q_now.device)
            energy_corr = torch.zeros(1, 1, device=q_now.device)

            if metal_index.sum() == 0:
                q_combined = q_now.clone()
                if self.adjust_neutrality:
                    # Adjust the charge of the electrolyte atoms to ensure neutrality
                    q_combined[electrolyte_index] = q_combined[electrolyte_index] - q_combined[electrolyte_index].sum() / electrolyte_index.sum()
                # If there are no metal atoms, we just return the original charges
                results.append(q_combined)
            else:
                # get the A matrix
                r = r_now[metal_index, :]
                B_ext = torch.zeros_like(r[:, self.external_field_direction])

                # if the positions of metal atoms haven't changed, we use the stored S matrix
                if self.S is None or self.A_mat is None or self.r.shape != r.shape  or not torch.allclose(self.r, r):
                    self.S = self._compute_S_matrix(r.detach(), cell.detach())
                    self.r = r.detach()

                if self.adjust_neutrality:
                    # Adjust the charge of the metal atoms to ensure neutrality
                    q_now[electrolyte_index] = q_now[electrolyte_index] - q_now[electrolyte_index].sum() / electrolyte_index.sum()

                _, f_now = self.ep.compute_potential_triclinic(r_now, 
                                                           q_now, 
                                                           cell, 
                                                           compute_field=True)

                if self.external_field is not None:
                    # assumes that the metal electrodes are on the sides, and the electrolytes are in the middle
                    # E = \dPhi / L
                    # energy_external = - E * r * q
                    if self.external_field_on:
                        energy_external = - self.external_field * torch.sum(r_now[electrolyte_index, self.external_field_direction].unsqueeze(1) * q_all[electrolyte_index]).unsqueeze(0) * self.external_field_norm_factor
                    if self.external_field_potential_on:
                        lz = cell[self.external_field_direction, self.external_field_direction]
                        r_wrap = r[:, self.external_field_direction] / lz
                        r_wrap = r_wrap - torch.round(r_wrap)
                        r_wrap = r_wrap * lz
                        B_ext = self.external_field * r_wrap * self.external_field_norm_factor
                # q = S^-1 * (- f - E * r)
                # the external field generated by electrolyte atoms needs to be scaled by \sqrt{\epsilon_r}
                B_mat = - f_now[metal_index, :] * self.scaling_factor  + B_ext.unsqueeze(1)

                q_mw = q_now.clone()
                # we then scale the true qs to get q^les
                q_mw[metal_index] = self.S @ B_mat / self.scaling_factor
                results.append(q_mw)

                # the electrostatic energy of the electrode will be estimated as q^les_i*q^les_j/r_ij = q_i*q_j / (r_ij \epsilon_infty),
                # but it's actually q_i*q_j / r_ij = (q^les_i*q^les_j/r_ij) * \epsilon_infty
                # so we calculate the difference
                if self.scaling_factor != 1.0:
                    energy_corr = q_mw[metal_index].T @ self.A_mat @ q_mw[metal_index] * (self.scaling_factor**2. - 1) / 2.
            energy_corr_results.append(energy_corr[0])
            energy_external_results.append(energy_external)


        data[self.output_key] = torch.cat(results, dim=0)
        data['mw_energy_correction'] = torch.cat(energy_corr_results, dim=0)
        data['mw_energy_external'] = torch.cat(energy_external_results, dim=0)
            
        data[self.output_key] = torch.cat(results, dim=0)
        
        return data
        

    def _compute_S_matrix(self, r, cell):
        N = len(r)
        q_eye = torch.eye(N, device=r.device)
        _, A_mat = self.ep.compute_potential_triclinic(r, q_eye, cell, compute_field=True)
        self.A_mat = A_mat
            
        A_inv = torch.inverse(A_mat)

        # Define E as a column vector of ones
        E = torch.ones(N, 1, device=r.device)

        # Compute the scalar denominator: E^T A^{-1} E
        denom = E.T @ A_inv @ E  # shape: (1, 1)

        # Compute the numerator: A^{-1} E E^T A^{-1}
        numer = A_inv @ E @ E.T @ A_inv  # shape: (N, N)

        # Final result: S = A^{-1} - numer / denom
        S = A_inv - numer / denom
        return S
