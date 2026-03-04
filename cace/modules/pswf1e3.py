"""PSWF long-range potential module (1e-3 error target variant)."""

import torch
import torch.nn as nn
from typing import Dict

__all__ = ["PSWFPotential_e3"]


class PSWFPotential_e3(nn.Module):
    def __init__(self,
                 dl = 2.0,  # grid resolution 
                 rcut = 4.0,
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
        self.rcut = rcut
        
        
        self.c = 9.5392
        self.C0 = 0.405792427033595 # \int_0^\inf psi0(x) dx
        self.C0_F = 3.14159231443538 # \int_0^\inf \hat{psi0}(k) dk
        self.Lambda = 0.811584854067189 # lambda parameter for PSWF
        self.phi0 = 1 # psi0(0)
        self.phi0_d2 = -8.76626080077216 # psi0''(0)
        self.phi0_d4 = 206.242436902525 # psi0''''(0)
        self.phi0_d6 = -7255.49767884584 # psi0''''''(0)
        
        # Fourier space kernel approximation for PSWF
        if exponent == 1:
            self.mono_coef = [
            2.00000025719292, -7.52257924281421e-05, -8.76261588292549,
            -0.0692064563692001, 17.8657483927947, -3.92467980783314,
            -5.82189849995526, -34.1528521528269, 69.1349621837278,
            -51.8399132153109, 17.9813979065892, -2.40933563358857
            ]
        elif exponent == 2:
            self.mono_coef = [
            3.14159176380425, -7.74165331161184, -0.00880935918071624,
            11.450104854176, -1.12081340855718, -8.05439915290295,
            -15.1816271757747, 38.6850032992473, -30.3065981127196,
            10.466045704118, -1.32884439689123
            ]
        elif exponent == 3:
            self.mono_coef = [
            -2.11125448144839, -0.000128868030097415, 4.38796571926236,
            -0.0677015201113482, -3.82966473710864, -1.79535190277462,
            7.38918785544925, -5.17075542809035, 1.19770494347521
            ]
        elif exponent == 4:
            self.mono_coef = [
            0.160088083353638, -1.82232585945277e-06, 0.785461964822818,
            -0.000820418306336654, -0.568942959493872, -0.0119466649524336,
            0.444817254538839, 0.109007656875178, -0.574420805150609,
            0.373882409312506, -0.0797074600524301
            ]
        elif exponent == 5:
            self.mono_coef = [
            -0.101915228480238, -1.57472310825861e-06, -0.555551723287083,
            -0.00142796548680263, 1.10966912781074, -0.0794847564415548,
            -0.429189480900115, -0.674454586950965, 1.45247036913404,
            -0.985036414321245, 0.298123420671996, -0.033201182608207
            ]
        elif exponent == 6:
            self.mono_coef = [
            0.00928380667501686, 4.12746752742916e-07, -0.152892824345803,
            0.000378707774465982, -0.37870911715477, 0.0214050015409513,
            0.104622115053315, 0.185436493936098, -0.395198670668172,
            0.279283664461165, -0.0907534323717394, 0.0113514142522663
            ]
        else:
            raise ValueError(f"Unsupported exponent value: {exponent}")
        
        # Real space kernel approximation for PSWF
        if exponent == 1:
            self.real_space_coef = [
            2.83227534889595e-07, 2.4642447844935, 0.00280410611886386,
            -3.64468157293283, 0.356765988554739, 2.56379487749303,
            4.83246201870804, -12.3138189975948, 9.64688979618854,
            -3.33144581701052, 0.422984308756931
            ]
        elif exponent == 2:
            self.real_space_coef = [
            -1.28596459013862e-07, 3.76128961957541e-05, 4.38130794146327,
            0.0346032281777482, -8.93287419635016, 1.96233990372563,
            2.91094925046335, 17.0764260756111, -34.5674810910013,
            25.9199566070716, -8.99069895306821, 1.20466781675587
            ]
        elif exponent == 3:
            self.real_space_coef = [
            -3.99899397239946e-07, 0.000111962487258865, -0.00514843639718487,
            7.2926416585047, -0.830181749174431, -12.618662002353,
            -13.670898773914, 47.5287345881554, -27.7974058129832,
            -8.99141286124119, 13.7437099199227, -3.65337584089231
            ]
        elif exponent == 4:
            self.real_space_coef = [
            -3.27299083896331e-06, 0.000767441819135906, -0.0294530920853255,
            0.434486387445452, 5.3800120640497, 13.3497846552701,
            -52.386453374095, 42.7394174149664, 3.05575968412511,
            -17.1214570264477, 5.56063345396331
            ]
        elif exponent == 5:
            self.real_space_coef = [
            2.30970088365215e-06, -0.000623671571050144, 0.0279199821529397,
            -0.488899726667487, 4.40053414959494, -11.8241666016335,
            75.4273603700678, -184.132904054476, 197.712971971958,
            -99.6341894075931, 19.4842432228787
            ]
        elif exponent == 6:
            self.real_space_coef = [
            4.19906566202083e-09, -2.11618762487042e-06, 0.000172227802999581,
            -0.00542194324998224, 0.0879434934627668, -0.846676710466824,
            15.3217940343921, -21.9250876116467, 39.5258190612743,
            -128.225052544905, 205.738735948899, -162.270298290063,
            63.4748916473441, -9.97578814171605
            ]
        else:
            raise ValueError(f"Unsupported exponent value: {exponent}")

        self.exponent = exponent
        
        
        self.twopi = 2.0 * torch.pi
        self.twopi_sq = self.twopi ** 2
        self.remove_self_interaction = remove_self_interaction
        self.feature_key = feature_key
        self.output_key = output_key
        self.aggregation_mode = aggregation_mode
        self.model_outputs = [output_key]
        # 1/2\epsilon_0, where \epsilon_0 is the vacuum permittivity
        # \epsilon_0 = 5.55263*10^{-3} e^2 eV^{-1} A^{-1}
        # self.norm_factor = 90.0474
        self.norm_factor = 1.0 
        # when using a norm_factor = 1, all "charges" are scaled by sqrt(90.0474)
        # the external field is then scaled by sqrt(90.0474) = 9.48933
        
        self.k_sq_max = (self.c/self.rcut)**2 #(self.twopi / self.dl) ** 2
        
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
        # print("Ewald box:",box)
        r = data['positions']
        q = data[self.feature_key]
        if q.dim() == 1:
            q = q.unsqueeze(1)
        #print("q_shape",q.dim(),q.size(0),q.shape) ##(2,82)

        # Check the input dimension 2
        n, d = r.shape
        #print("r_shape",n,d) ##(82,3)
        assert d == 3, 'r dimension error'
        assert n == q.size(0), 'q dimension error'

        unique_batches = torch.unique(batch_now)  # Get unique batch indices

        results = []
        field_results = []
        for i in unique_batches:
            mask = batch_now == i  # Create a mask for the i-th configuration
            # Calculate the potential energy for the i-th configuration
            r_raw_now, q_now, box_now = r[mask], q[mask], box[i]
            #print(r_raw_now.shape, q_now.shape, box_now.shape)
            box_diag = box[i].diagonal(dim1=-2, dim2=-1)
            if box_diag[0] < 1e-6 and box_diag[1] < 1e-6 and box_diag[2] < 1e-6:
                pot = self.compute_potential_realspace(r_raw_now, q_now)
            elif box_diag[0] > 0 and box_diag[1] > 0 and box_diag[2] > 0:
                # the box is periodic, we use the reciprocal sum
                pot = self.compute_potential_triclinic(r_raw_now, q_now, box_now)
            else:
                raise ValueError("Either all box dimensions must be positive or aperiodic box must be provided.")

            if hasattr(self, 'external_field') and self.external_field is not None:
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

        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_realspace(self, r_raw, q):
        # Compute pairwise distances (norm of vector differences)
        r_ij = r_raw.unsqueeze(0) - r_raw.unsqueeze(1)
        r_ij_norm = torch.norm(r_ij, dim=-1)
        #print(r_ij_norm)
        n = r_ij_norm.size(0)
        idx = torch.arange(n, device=r_ij_norm.device)
        # Error function scaling for long-range interactions

        #convergence_func_ij = torch.special.erf(r_ij_norm / self.sigma / (2.0 ** 0.5)) # Ewald
        input_rfactor = r_ij_norm / self.rcut
        convergence_func_ij = torch.zeros_like(r_ij_norm)
        for coef in reversed(self.real_space_coef):
            convergence_func_ij = convergence_func_ij * input_rfactor + coef
        convergence_func_ij = torch.where(input_rfactor > 1, torch.ones_like(convergence_func_ij), convergence_func_ij)
        
        convergence_func_ij[idx, idx] = 0.0 
        
        if q.dim() == 1:
            # [n_node, n_q]
            q = q.unsqueeze(1)
   
        # Compute inverse distance safely
        # [n_node, n_node]
        #r_p_ij = torch.where(r_ij_norm > 1e-3, 1.0 / r_ij_norm, 0.0) # this causes gradient issues
        epsilon = 1e-6
        
        if self.exponent == 1:
            r_p_ij = 1.0 / (r_ij_norm + epsilon)
        elif self.exponent == 2:
            r_p_ij = 1.0 / (r_ij_norm**2 + epsilon)
        elif self.exponent == 3:
            r_p_ij = 1.0 / (r_ij_norm**3 + epsilon)
        elif self.exponent == 4:
            r_p_ij = 1.0 / (r_ij_norm**4 + epsilon)
        elif self.exponent == 5:
            r_p_ij = 1.0 / (r_ij_norm**5 + epsilon)
        elif self.exponent == 6:
            r_p_ij = 1.0 / (r_ij_norm**6 + epsilon)
    
        # Compute potential energy
        n_node, n_q = q.shape
        # Use broadcasting to set diagonal elements to 0
        #mask = torch.ones(n_node, n_node, n_q, dtype=torch.int64, device=q.device)
        #diag_indices = torch.arange(n_node)
        #mask[diag_indices, diag_indices, :] = 0
        # [1, n_node, n_q] * [n_node, 1, n_q] * [n_node, n_node, 1] * [n_node, n_node, 1]
        pot = torch.sum(q.unsqueeze(0) * q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2)).view(-1) / self.twopi / 2.0
        
        #print(convergence_func_ij[0,0],convergence_func_ij[1,1])
        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            if self.exponent == 1:
                #pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.)) # Ewald
                pot += torch.sum(q ** 2) * self.phi0 / (self.twopi * 2. * self.C0 * self.rcut)
                #q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2. # Ewald
                #q_field = q_field + q * self.phi0 / (self.twopi * self.C0 * self.rcut)
            elif self.exponent == 2:
                pot += torch.sum(q ** 2) * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / (self.twopi * 2.)
            elif self.exponent == 3:
                pot += torch.sum(q ** 2) * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / (self.twopi * 2.)
            elif self.exponent == 4:
                pot += torch.sum(q ** 2) * (self.phi0_d4/(24. * (self.rcut**4))) / (self.twopi * 2.)
            elif self.exponent == 5:
                pot += torch.sum(q ** 2) * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / (self.twopi * 2.)
            elif self.exponent == 6:
                pot += torch.sum(q ** 2) * (-self.phi0_d6/(720. * (self.rcut**6))) / (self.twopi * 2.)
    
        return pot * self.norm_factor
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, box):
        device = r_raw.device

        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T  # Reciprocal lattice vectors [3,3], G = 2π(M^{-1}).T

        # max Nk for each axis
        norms = torch.norm(box, dim=1)
        Nk = [max(1, int(n.item() / self.dl)) for n in norms]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, device=device)
        
        #print(G)
        #print(cell_inv.T)
        #print("The number of Fourier grids is ", Nk) # 6

        # Create nvec grid and compute k vectors
        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        # kvec = G @ nvec
        kvec = (nvec.float() @ G).to(r_raw.device)  # [N_total, 3]

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec ** 2, dim=1)
        #mask = (k_sq > 0) & (k_sq <= self.k_sq_max)
        mask = (k_sq <= self.k_sq_max)
        if self.exponent in (1, 2, 3):
            mask = mask & (k_sq > 0)   # remove k=0 only at exponent=1/2/3

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
        if q.dim() == 1:  
            q = q.unsqueeze(1)
         #for torchscript compatibility, to avoid dtype mismatch, only use real part
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)
        S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        S_k_sq = S_k_real**2 + S_k_imag**2  # [M]
 
        # Compute kfac,  exp(-σ^2/2 k^2) / k^2 for exponent = 1
        k_abs = torch.sqrt(k_sq)
        input_kfactor = k_abs * self.rcut / self.c
        is_k0 = (k_sq == 0)
        kfac = torch.zeros_like(input_kfactor)
        if self.exponent == 1:
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
            kfac = kfac / k_sq
            # kfac = torch.exp(-self.sigma_sq_half * k_sq) / k_sq
        elif self.exponent == 2:
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
            kfac = kfac / (k_abs)
        elif self.exponent == 3:
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = kfac - self.Lambda / self.C0 * self.phi0 * torch.log(input_kfactor)
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
        elif self.exponent == 4:
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = -torch.pi / 2. * k_abs + kfac * (self.c**2) * self.Lambda / (self.C0_F * self.rcut)
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
        elif self.exponent == 5:
            # poly = mono(input_kfactor)
            poly = torch.zeros_like(input_kfactor)
            for coef in reversed(self.mono_coef):
                poly = poly * input_kfactor + coef
            
            term2 = - poly * (self.c**2) * self.Lambda / (6 * self.C0 * self.rcut**2)
            log_x = torch.zeros_like(input_kfactor)
            nonzero = ~is_k0
            log_x[nonzero] = torch.log(input_kfactor[nonzero])
            term1 = k_sq * log_x * (self.Lambda * self.phi0) / (6 * self.C0)
            term1 = torch.where(is_k0, torch.zeros_like(term1), term1)
            
            kfac = term1 + term2
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
        elif self.exponent == 6:
            for coef in reversed(self.mono_coef):
                kfac = kfac * input_kfactor + coef
            kfac = (torch.pi / 24.) * k_sq * k_abs + kfac * (self.c**4) * self.Lambda * torch.pi / (24. * self.rcut**3 * self.C0_F)
            kfac = torch.where(input_kfactor > 1, torch.zeros_like(kfac), kfac)
            # print(k_abs, kfac)
        
        # Compute potential, (2π/volume)* sum(factors * kfac * |S(k)|^2)
        volume = torch.det(box)
        #pot0 = (factors * kfac * torch.abs(S_k0)**2).sum() / volume
        #print("pot0",pot0.shape,pot0)
        
        #pot = (factors * kfac * torch.abs(S_k)**2).sum() / volume  # Ewald
        pot = (factors * kfac * S_k_sq).sum(dim=1) / (2 * volume)
        pot.view(-1)
        #print("pot",pot.shape)

        # print(self.remove_self_interaction) # fauls
        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            if self.exponent == 1:
                #pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.)) # Ewald
                pot -= torch.sum(q ** 2) * self.phi0 / (self.twopi * 2. * self.C0 * self.rcut)
                #q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2. # Ewald
                #q_field -= q * self.phi0 / (self.twopi * self.C0 * self.rcut)
            elif self.exponent == 2:
                pot -= torch.sum(q ** 2) * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / (self.twopi * 2.)
                #q_field -= q * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / self.twopi
            elif self.exponent == 3:
                pot -= torch.sum(q ** 2) * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / (self.twopi * 2.)
                #q_field -= q * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / self.twopi
            elif self.exponent == 4:
                pot -= torch.sum(q ** 2) * (self.phi0_d4/(24. * (self.rcut**4))) / (self.twopi * 2.)
                #q_field -= q * (self.phi0_d4/(24. * (self.rcut**4))) / self.twopi
            elif self.exponent == 5:
                pot -= torch.sum(q ** 2) * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / (self.twopi * 2.)
                #q_field -= q * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / self.twopi
            elif self.exponent == 6:
                pot -= torch.sum(q ** 2) * (-self.phi0_d6/(720. * (self.rcut**6))) / (self.twopi * 2.)
                #q_field -= q * (-self.phi0_d6/(720. * (self.rcut**6))) / self.twopi
        
        return pot * self.norm_factor
    
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
