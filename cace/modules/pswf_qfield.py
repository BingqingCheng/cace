"""PSWF long-range potential module with optional field output."""

import torch
import torch.nn as nn
from typing import Dict

__all__ = ["PSWFPotential_Qfield"]


class PSWFPotential_Qfield(nn.Module):
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
        
        
        self.c = 12.024
        self.C0 = 0.361439368296059 # \int_0^\inf psi0(x) dx
        self.C0_F = 3.14159265091592 # \int_0^\inf \hat{psi0}(k) dk
        self.Lambda = 0.722878736592121 # lambda parameter for PSWF
        self.phi0 = 1 # psi0(0)
        self.phi0_d2 = -11.2564605886249 # psi0''(0)
        self.phi0_d4 = 348.322293451646 # psi0''''(0)
        self.phi0_d6 = -16483.4766568324 # psi0''''''(0)
        
        # Fourier space kernel approximation for PSWF
        if exponent == 1:
            self.mono_coef = [
            1.99999288998575, 0.00168787785084523, -11.3222097075807,
            0.987921696028683, 21.5457341750713, 32.0951969525458,
            -127.138762636063, 118.131826030792, -29.1811517575215,
            -13.5720497316504, 6.45196584189916
            ]
        elif exponent == 2:
            self.mono_coef = [
            3.14159075044132, -8.69140523747556, -0.0208341314298224,
            16.654582626159, -2.99107582260391, -10.1973605520683,
            -46.9943145751952, 121.536979999833, -113.420998450019,
            49.5334410010207, -8.55060507643342
            ]
        elif exponent == 3:
            self.mono_coef = [
            -2.34895982222418, -0.000123674713465791, 5.63294919731762,
            -0.0690238299506774, -6.7529214045764, -2.04627823592562,
            12.355932854338, -5.55975770714042, -5.2638772643861,
            5.44279485784447, -1.390735615535
            ]
        elif exponent == 4:
            self.mono_coef = [
            0.127795201214023, 8.12764442611525e-07, 0.785353341709381,
            0.000966088346860502, -0.74747066863158, 0.0702136632863331,
            0.470398783554688, 0.776632828484982, -1.99336937282461,
            1.44101235857374, -0.314432598714449, -0.088537950494638,
            0.0391851473150068
            ]
        elif exponent == 5:
            self.mono_coef = [
                -0.0813569536157143, 6.81583155122704e-08, -0.674484997821879,
                0.000134872078079662, 1.40508108190434, 0.0174223322512818,
                -1.30934573084683, 0.390084183214956, -0.109370966667372,
                2.03608900531881, -3.3215140477848, 2.35234372838666,
                -0.822214060747275, 0.117131486198371
            ]
        elif exponent == 6:
            self.mono_coef = [
                0.00606279899039218, -7.63739851558903e-09, -0.122034844272473,
                -1.76299076025651e-05, -0.374723920116058, -0.0025858504250432,
                0.250187640249132, -0.0644704437589401, 0.00296447981119563,
                -0.369282586996706, 0.633175709300539, -0.462855498334648,
                0.166552701005677, -0.0244119162282381
            ]
        else:
            raise ValueError(f"Unsupported exponent value: {exponent}")
        
        # Real space kernel approximation for PSWF
        if exponent == 1:
            self.real_space_coef = [
                -1.07538610903007e-08, 2.76671977587222, -0.000248133153035152,
                -5.18451117976801, -0.0771836300678381, 8.61808700109746,
                -2.87598455665560, 0.402380987157713, -21.1916233369615,
                39.8233819429679, -31.5480841045135, 12.1563446112467,
                -1.88927936977675
            ]
        elif exponent == 2:
            self.real_space_coef = [
                1.56872217668171e-09, -8.54271046223445e-07, 5.62830359164959,
                -0.00239932685066702, -14.47330153481, -0.396208345763171,
                25.4019471927678, -10.6911391017736, 6.61183206403194,
                -64.7520941810229, 111.474851965543, -84.5622766804132,
                31.5187206345318, -4.75830754682796
            ]
        elif exponent == 3:
            self.real_space_coef = [
                4.67939518448805e-09, -1.94150660242157e-06, 0.000130892677335293,
                10.3777647957088, 0.0434928706284481, -32.4336139421979,
                1.20418585997371, 52.6312984951602, -6.88184306903349,
                -17.0868298175951, -120.245636630316, 245.837561336938,
                -195.452464310416, 74.3419714387119, -11.3362155149836
            ]
        elif exponent == 4:
            self.real_space_coef = [
            4.82988312218088e-08, -1.7436895316511e-05, 0.00102058558760565,
            -0.0227500783644743, 14.7641198279769, -1.49957919213718,
            -41.1593022634368, -2.94782031226714, 42.4834067447795,
            137.023449598545, -359.120281938347, 326.434166566877,
            -137.993574733502, 23.0346869574518
            ]
        elif exponent == 5:
            self.real_space_coef = [
            -2.58077172346147e-07, 0.000103020421970718, -0.00682225052726304,
            0.177953368535304, -2.42065564495033, 41.1053491138045,
            -103.37210236923, 293.354715484923, -889.610288179988,
            1606.97883781275, -1645.1388954413, 963.136161858937,
            -303.215515995304, 40.0065264083304
            ]
        elif exponent == 6:
            self.real_space_coef = [
            -2.7742656225592e-07, 0.000105486371883429, -0.00660354980891142,
            0.16125065349048, -2.02710638746153, 14.970060788007,
            -46.6465679746217, 209.060770179616, -476.868207135693,
            453.326283342698, -66.2007852190316, -194.190986421974,
            139.368017193798, -29.9689844903318
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
        #self.norm_factor = 90.0474
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

        #print(results[0].shape,results[1].shape,results[2].shape, pot.shape)
        data[self.output_key] = torch.stack(results, dim=0).sum(axis=1) if self.aggregation_mode == "sum" else torch.stack(results, dim=0)
        if self.compute_field:
            data[self.feature_key+'_field'] = torch.cat(field_results, dim=0)
        return data

    def compute_potential_realspace(self, r_raw, q, compute_field=False):
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
        epsilon = 1e-8
        
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
        
        q_field = torch.zeros_like(q, dtype=q.dtype, device=q.device)
        if compute_field:
            q_field = torch.sum(q.unsqueeze(1) * r_p_ij.unsqueeze(2) * convergence_func_ij.unsqueeze(2), dim=0) / self.twopi
        
        # because this realspace sum already removed self-interaction, we need to add it back if needed
        if self.remove_self_interaction == False:
            if self.exponent == 1:
                #pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.)) # Ewald
                pot += torch.sum(q ** 2) * self.phi0 / (self.twopi * 2. * self.C0 * self.rcut)
                #q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2. # Ewald
                q_field = q_field + q * self.phi0 / (self.twopi * self.C0 * self.rcut)
            elif self.exponent == 2:
                pot += torch.sum(q ** 2) * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / (self.twopi * 2.)
                q_field = q_field + q * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / (self.twopi)
            elif self.exponent == 3:
                pot += torch.sum(q ** 2) * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / (self.twopi * 2.)
                q_field = q_field + q * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / (self.twopi)
            elif self.exponent == 4:
                pot += torch.sum(q ** 2) * (self.phi0_d4/(24. * (self.rcut**4))) / (self.twopi * 2.)
                q_field = q_field + q * (self.phi0_d4/(24. * (self.rcut**4))) / (self.twopi)
            elif self.exponent == 5:
                pot += torch.sum(q ** 2) * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / (self.twopi * 2.)
                q_field = q_field + q * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / (self.twopi)
            elif self.exponent == 6:
                pot += torch.sum(q ** 2) * (-self.phi0_d6/(720. * (self.rcut**6))) / (self.twopi * 2.)
                q_field = q_field + q * (-self.phi0_d6/(720. * (self.rcut**6))) / (self.twopi)
    
        return pot * self.norm_factor, q_field * self.norm_factor
    
    # Triclinic box(could be orthorhombic)
    def compute_potential_triclinic(self, r_raw, q, box, compute_field=False):
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
        
        exp_ikr = torch.exp(1j * k_dot_r)
        S_k = (q.unsqueeze(2) * exp_ikr.unsqueeze(1)).sum(dim=0)
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
            #kfac = k_sq * torch.log(input_kfactor) * self.Lambda * self.phi0 / (6 * self.C0) - kfac * (self.c**2) * self.Lambda / (6 * self.C0 * self.rcut**2)
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
        
        # Compute electric field if needed
        q_field = torch.zeros_like(q, dtype=r_raw.dtype, device=device)
        if compute_field:
            sk_field = 2 * kfac * torch.conj(S_k)                               # [n_q, M]
            q_field = (factors * torch.real(exp_ikr.unsqueeze(1)                # [N,1,M]
                                            * sk_field.unsqueeze(0)))           # [1,n_q,M]
            q_field = q_field.sum(dim=2) / (2 * volume)                               # [N,n_q]

        # print(self.remove_self_interaction) # fauls
        # Remove self-interaction if applicable
        if self.remove_self_interaction:
            if self.exponent == 1:
                #pot += torch.sum(q ** 2) / (self.sigma * self.twopi**(3./2.)) # Ewald
                pot -= torch.sum(q ** 2) * self.phi0 / (self.twopi * 2. * self.C0 * self.rcut)
                #q_field = q_field + q / (self.sigma * self.twopi**(3./2.)) * 2. # Ewald
                q_field -= q * self.phi0 / (self.twopi * self.C0 * self.rcut) # ESP
            elif self.exponent == 2:
                pot -= torch.sum(q ** 2) * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / (self.twopi * 2.)
                q_field -= q * (-self.phi0_d2/(2. * self.rcut * self.rcut)) / self.twopi
            elif self.exponent == 3:
                pot -= torch.sum(q ** 2) * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / (self.twopi * 2.)
                q_field -= q * ( -self.phi0_d2/(3. * (self.rcut**3) * self.C0)) / self.twopi
            elif self.exponent == 4:
                pot -= torch.sum(q ** 2) * (self.phi0_d4/(24. * (self.rcut**4))) / (self.twopi * 2.)
                q_field -= q * (self.phi0_d4/(24. * (self.rcut**4))) / self.twopi
            elif self.exponent == 5:
                pot -= torch.sum(q ** 2) * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / (self.twopi * 2.)
                q_field -= q * (self.phi0_d4/(45. * (self.rcut**5) * self.C0)) / self.twopi
            elif self.exponent == 6:
                pot -= torch.sum(q ** 2) * (-self.phi0_d6/(720. * (self.rcut**6))) / (self.twopi * 2.)
                q_field -= q * (-self.phi0_d6/(720. * (self.rcut**6))) / self.twopi
        
        return pot * self.norm_factor, q_field * self.norm_factor
    
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
