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
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z\
                 feature_key: str = 'q',
                 output_key: str = 'q_mw',
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
        
        self.feature_key = feature_key
        self.output_key = output_key
        
        self.model_outputs = [output_key]
        
    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        
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
            q_all = q.unsqueeze(1)

        # Check the input dimension
        n, d = r_all.shape
        assert d == 3, 'r dimension error'
        assert n == q_all.size(0), 'q dimension error'
        
        unique_batches = torch.unique(batch_now)  # Get unique batch indices
                    
        results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_now, q_now, atomic_numbers, cell = r_all[mask], q_all[mask], atomic_numbers_all[mask], box_all[i]
            
            metal_index = (atomic_numbers == self.metal_atomic_numbers)
            electrode_index = ~metal_index
            
        
            # get the A matrix
            r = r_now[metal_index, :]
            cell = data['cell']

            # if the positions of metal atoms haven't changed, we use the stored S matrix
            if self.S is None or not torch.allclose(self.r, r):
                self.S = self._compute_S_matrix(r.detach(), cell.detach())
                self.r = r.detach()


            q_combined = q_now.clone()
            q_combined[metal_index] = 0.0
            _, f_now = self.ep.compute_potential_triclinic(r_now, 
                                                           q_combined, 
                                                           cell, 
                                                           compute_field=True)
            B_mat = f_now[metal_index, :] * -1.

            q_mw = q_combined.clone()
            q_mw[metal_index] = self.S @ B_mat
            results.append(q_mw)
            
        data[self.output_key] = torch.cat(results, dim=0)
        
        return data
        


    def _compute_S_matrix(self, r, cell):
        N = len(r)
        #A_mat = torch.zeros((N, N), device=r.device)
        #for i in range(N):
        #    q_trail = torch.zeros(N)
        #    q_trail[i] = 1.
        #    _, f_now = self.ep.compute_potential_triclinic(r, q_trail.unsqueeze(1), cell, compute_field=True)
        #    A_mat[i, :] = f_now[:, 0]
        q_eye = torch.eye(N, device=r.device)
        box = cell.view(3, 3).diagonal(dim1=-2, dim2=-1)
        _, A_mat = self.ep.compute_potential_optimized(r, q_eye, box, compute_field=True)
        #_, A_mat = self.ep.compute_potential_triclinic(r, q_eye, cell, compute_field=True)
            
        A_inv = torch.inverse(A_mat)

        # Define E as a column vector of ones
        E = torch.ones(N, 1)

        # Compute the scalar denominator: E^T A^{-1} E
        denom = E.T @ A_inv @ E  # shape: (1, 1)

        # Compute the numerator: A^{-1} E E^T A^{-1}
        numer = A_inv @ E @ E.T @ A_inv  # shape: (N, N)

        # Final result: S = A^{-1} - numer / denom
        S = A_inv - numer / denom
        return S
