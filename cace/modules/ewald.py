import torch
import torch.nn as nn
from itertools import product
from typing import Dict
import numpy as np

class EwaldPotential(nn.Module):
    def __init__(self,
                 dl=2.0,  # grid resolution
                 sigma=1.0,  # width of the Gaussian on each atom
                 exponent=1, # default is for electrostattics with p=1, we can do London dispersion with p=6
                 external_field = None, # external field
                 external_field_direction: int = 0, # external field direction, 0 for x, 1 for y, 2 for z
                 charge_neutral_lambda: float = None,
                 remove_self_interaction=False,
                 feature_key: str = 'q',
                 output_key: str = 'ewald_potential',
                 aggregation_mode: str = "sum",
                 compute_field: bool = False,
                 ):
        super().__init__()
        self.dl = dl
        self.sigma = sigma
        self.exponent = exponent
        self.sigma_sq_half = sigma ** 2 / 2.0
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        #self.norm_factor = 90.0474
        self.norm_factor = 1.0 
        # when using a norm_factor = 1, all "charges" are scaled by sqrt(90.0474)
        # the external field is then scaled by sqrt(90.0474) = 9.48933
        self.k_sq_max = (self.twopi / self.dl) ** 2
        self.external_field = external_field
        self.external_field_direction = external_field_direction
        self.compute_field = compute_field
        if self.compute_field:
            self.model_outputs.append(feature_key+'_field')

        self.charge_neutral_lambda = charge_neutral_lambda

    def forward(self, data: Dict[str, torch.Tensor], **kwargs):
        if data["batch"] is None:
            n_nodes = data['positions'].shape[0]
            batch_now = torch.zeros(n_nodes, dtype=torch.int64, device=data['positions'].device)
        else:
            batch_now = data["batch"]

        # this is just for compatibility with the previous version
        if hasattr(self, 'exponent') == False:
            self.exponent = 1
        if hasattr(self, 'compute_field') == False:
            self.compute_field = False
        
        # box = data['cell'].view(-1, 3, 3).diagonal(dim1=-2, dim2=-1)
        box = data['cell'].view(-1, 3, 3)
        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)

        # Check the input dimension
        n, d = r.shape
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        field_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i]
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6 and self.exponent == 1:
                # the box is not periodic, we use the direct sum
                pot, field = self.compute_potential_realspace(r_raw_now, q_now, self.compute_field)
            elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
                # the box is periodic, we use the reciprocal sum
                pot, field = self.compute_potential_triclinic(r_raw_now, q_now, box_now, self.compute_field)
            else:
                raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")

            if self.exponent == 1 and hasattr(self, 'external_field') and self.external_field is not None:
                # if self.external_field_direction is an integer, then external_field_direction is the direction index
                if isinstance(self.external_field_direction, int):
                    direction_index_now = self.external_field_direction
                    # if self.external_field_direction is a string, then it is the key to the external field
                else:
                    try:
                        direction_index_now = int(data[self.external_field_direction][i])
                    except:
                        raise ValueError("external_field_direction must be an integer or a key to the external field")
                if isinstance(self.external_field, float):
                    external_field_now = self.external_field
                else:
                    try:
                        external_field_now = data[self.external_field][i]
                    except:
                        raise ValueError("external_field must be a float or a key to the external field")
                box_now = box_now.diagonal(dim1=-2, dim2=-1)
                pot_ext = self.add_external_field(r_raw_now, q_now, box_now, direction_index_now, external_field_now)
            else:
                pot_ext = 0.0

            if hasattr(self, 'charge_neutral_lambda') and self.charge_neutral_lambda is not None:
                q_mean = torch.mean(q[mask])
                pot_neutral = self.charge_neutral_lambda * (q_mean)**2.
                #print(pot_neutral, pot)
            else:
                pot_neutral = 0.0

            results.append(pot + pot_ext + pot_neutral)
            field_results.append(field)

        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_realspace(self, r_raw, q, compute_field=False):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
 
        # Error function scaling for long-range interactions
        convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5))
        #print(convergence_func_ij)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        r_p_ij = 1.0 / (r_ij_norm + epsilon)

        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
    
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device) # Field due to q
        # Compute field if requested
        if compute_field:
            # [n_node, 1 , n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi

        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False and self.exponent == 1:
            pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2.
    
        return pot * self.norm_factor, q_field * self.norm_factor
 

    def compute_potential(self, r_raw, q, box, compute_field=False):
        """ Compute the Ewald long-range potential for one configuration """
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device


        volume = box[0] * box[1] * box[2]
        r = r_raw / box  # Work with scaled positions
        # r =  r - torch.round(r) # periodic boundary condition

        # Calculate nk based on the provided box dimensions and resolution
        nk = (box / self.dl).int().tolist()
        for i in range(3):
            if nk[i] < 1: nk[i] = 1
        n = r.shape[0]
        eikx = torch.zeros((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.zeros((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.zeros((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 0] = torch.ones(n, dtype=dtype, device=device)
        eiky[:, nk[1]] = torch.ones(n, dtype=dtype, device=device)
        eikz[:, nk[2]] = torch.ones(n, dtype=dtype, device=device)

        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(1, nk[0] + 1):
            eikx[:, k] = torch.exp(1j * self.twopi * k * r[:, 0]) 
        for k in range(1, nk[1] + 1):
            eiky[:, nk[1] + k] = torch.exp(1j * self.twopi * k * r[:, 1])
        for k in range(1, nk[2] + 1):
            eikz[:, nk[2] + k] = torch.exp(1j * self.twopi * k * r[:, 2])

        # Negative k values are complex conjugates of positive ones
        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        pot_list = []
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device) # Field due to q
        
        for kx in range(nk[0] + 1):
            # for negative kx, the Fourier transform is just the complex conjugate of the positive kx
            factor = 1.0 if kx == 0 else 2.0

            for ky, kz in product(range(-nk[1], nk[1] + 1), range(-nk[2], nk[2] + 1)):
                k_sq = self.twopi_sq * ((kx / box[0]) ** 2 + (ky / box[1]) ** 2 + (kz / box[2]) ** 2)
                if k_sq <= self.k_sq_max and k_sq > 0:  # remove the k=0 term
                    if self.exponent == 1:
                        kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
                    elif self.exponent == 6:
                        b_sq = k_sq * self.sigma_sq_half
                        b = torch.sqrt(b_sq)
                        kfac = -1.0 * k_sq**(3/2) * (torch.pi**0.5 * torch.special.erfc(b) + (1 / (2 * b**3) - 1 / b) * torch.exp(-b_sq))
                    eik = (eikx[:, kx] * eiky[:, nk[1] + ky] * eikz[:, nk[2] + kz]).unsqueeze(1) # [n, 1]
                    sk = torch.sum(q * eik, dim=0) # [n_q]
                    sk_conj = torch.conj(sk)
                    sk_field = 2. * kfac * sk_conj # the factor of 2 comes from normalization factor 2\epsilon
                    pot_list.append(factor * kfac * torch.real(sk * sk_conj))
                    if compute_field:
                        # The reverse transform to get the real-space potential field
                        q_field += factor * torch.real(sk_field.unsqueeze(0) * eik)

        pot = torch.stack(pot_list).sum(axis=0) / volume
        if compute_field:
            q_field /= volume
        #print(pot, torch.sum(q * q_field, dim=0) /2) #should be the same

        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field - q / (self.sigma * self.twopi**(3./2.)) * 2.

        return pot.real * self.norm_factor, q_field * self.norm_factor

    # Optimized function
    def compute_potential_optimized(self, r_raw, q, box, compute_field=False):
        dtype = torch.complex64 if r_raw.dtype == torch.float32 else torch.complex128
        device = r_raw.device

        volume = box[0] * box[1] * box[2]
        r = r_raw / box

        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device) # Field due to q

        nk = (box / self.dl).int().tolist()
        nk = [max(1, k) for k in nk]

        n = r.shape[0]
        eikx = torch.ones((n, nk[0] + 1), dtype=dtype, device=device)
        eiky = torch.ones((n, 2 * nk[1] + 1), dtype=dtype, device=device)
        eikz = torch.ones((n, 2 * nk[2] + 1), dtype=dtype, device=device)

        eikx[:, 1] = torch.exp(1j * self.twopi * r[:, 0])
        eiky[:, nk[1] + 1] = torch.exp(1j * self.twopi * r[:, 1])
        eikz[:, nk[2] + 1] = torch.exp(1j * self.twopi * r[:, 2])
        # Calculate remaining positive kx, ky, and kz terms by recursion
        for k in range(2, nk[0] + 1):
            eikx[:, k] = eikx[:, k - 1].clone() * eikx[:, 1].clone()
        for k in range(2, nk[1] + 1):
            eiky[:, nk[1] + k] = eiky[:, nk[1] + k - 1].clone() * eiky[:, nk[1] + 1].clone()
        for k in range(2, nk[2] + 1):
            eikz[:, nk[2] + k] = eikz[:, nk[2] + k - 1].clone() * eikz[:, nk[2] + 1].clone()

        # Negative k values are complex conjugates of positive ones
        for k in range(nk[1]):
            eiky[:, k] = torch.conj(eiky[:, 2 * nk[1] - k])
        for k in range(nk[2]):
            eikz[:, k] = torch.conj(eikz[:, 2 * nk[2] - k])

        kx = torch.arange(nk[0] + 1, device=device)
        ky = torch.arange(-nk[1], nk[1] + 1, device=device)
        kz = torch.arange(-nk[2], nk[2] + 1, device=device)

        kx_term = (kx / box[0]) ** 2
        ky_term = (ky / box[1]) ** 2
        kz_term = (kz / box[2]) ** 2

        kx_sq = kx_term.view(-1, 1, 1)
        ky_sq = ky_term.view(1, -1, 1)
        kz_sq = kz_term.view(1, 1, -1)

        k_sq = self.twopi_sq * (kx_sq + ky_sq + kz_sq) # [nx, ny, nz]

        if self.exponent == 1:
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq # [nx, ny, nz]
        elif self.exponent == 6:
            # Calculate b_sq and b
            b_sq = k_sq * self.sigma_sq_half
            b = torch.sqrt(b_sq)

            # Compute kfac based on the provided expression
            kfac = -1.0 * k_sq ** (3 / 2) * ( torch.pi ** 0.5 * torch.special.erfc(b) + (1 / (2 * b ** 3) - 1 / b) * torch.exp(-b_sq))
            #kfac = -1.0 * k_sq ** (3 / 2) * torch.exp(-b_sq) # this assumed a Gaussian smearing

        mask = (k_sq <= self.k_sq_max) & (k_sq > 0)
        kfac[~mask] = 0

        eikx_expanded = eikx.unsqueeze(2).unsqueeze(3) #[n_node, n_x, 1, 1]
        eiky_expanded = eiky.unsqueeze(1).unsqueeze(3) #[n_node, 1, n_y, 1]
        eikz_expanded = eikz.unsqueeze(1).unsqueeze(2) #[n_node, 1, 1, n_z]

        factor = torch.ones_like(kx, dtype=r_raw.dtype, device=device)
        factor[1:] = 2.0

        if q.dim() == 1:
            # [n_node, n_q, 1, 1, 1]
            q_expanded = q.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)
        elif q.dim() == 2:
            q_expanded = q.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        else:
            raise ValueError("q must be 1D or 2D tensor")
        # q_expanded: [n_node, n_q, 1, 1, 1]
        # eik: [n_node, n_x, n_y, n_z]
        # sk: [n_q, n_x, n_y, n_z]
        # kfac: [n_x, n_y, n_z]
        eik = eikx_expanded * eiky_expanded * eikz_expanded
        sk = torch.sum(q_expanded * eik.unsqueeze(1), dim=[0])
        sk_conj = torch.conj(sk)
        pot = (kfac.unsqueeze(0) * factor.view(1, -1, 1, 1) * torch.real(sk_conj * sk)).sum(dim=[1, 2, 3])
        # The reverse transform to get the real-space potential field
        if compute_field:
            sk_field = 2. * kfac.unsqueeze(0) * sk_conj  # the factor of 2 comes from normalization factor 2\epsilon
            q_field = (factor.view(1, 1, -1, 1, 1) * torch.real(eik.unsqueeze(1) * sk_field.unsqueeze(0))).sum(dim=[2, 3, 4])
            q_field /= volume

        pot /= volume

        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.))
            q_field = q_field - q / (self.sigma * self.twopi**(3./2.)) * 2.

        return pot.real * self.norm_factor, q_field * self.norm_factor

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
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, cell_now, compute_field=False):
        device = r_raw.device

        cell_inv = torch.linalg.inv(cell_now)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # max Nk for each axis
        norms = torch.norm(cell_now, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        # kvec = G @ nvec
        kvec = (nvec.float() @ G).to(device)  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        kvec = kvec[mask] # [M, 3]
        k_sq = k_sq[mask] # [M]
        nvec = nvec[mask] # [M, 3]

        # Determine symmetry factors (handle hemisphere to avoid double-counting)
        # Include nvec if first non-zero component is positive
        non_zero = (nvec != 0).to(torch.int)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0)

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(r_raw, kvec.T)  # [n, M]
        exp_ikr = torch.exp(1j * k_dot_r)
        S_k = torch.sum(q * exp_ikr, dim=0)  # [M]

        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        if self.exponent == 1:
            kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        elif self.exponent == 6:
            b_sq = k_sq * self.sigma_sq_half
            b = torch.sqrt(b_sq)
            kfac = -1.0 * k_sq**(3/2) * (
                torch.sqrt(torch.tensor(torch.pi)) * torch.special.erfc(b) + 
                (1/(2*b**3) - 1/b) * torch.exp(-b_sq)
            )
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(cell_now)
        pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume
        
        # Compute electric field if needed
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)
            q_field = (factors * torch.real(exp_ikr * sk_field)).sum(dim=1) / volume

        # Remove self-interaction if applicable
        if self.remove_self_interaction and self.exponent == 1:
            pot -= torch.sum(q**2) / (self.sigma * (2 * torch.pi)**1.5)
            q_field -= q * (2 / (self.sigma * (2 * torch.pi)**1.5))

        return pot.unsqueeze(0) * self.norm_factor, q_field.unsqueeze(1) * self.norm_factor
