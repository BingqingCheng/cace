import torch
import cace
from cace.modules import EwaldPotential

ep = EwaldPotential(dl=3,
                    sigma=3,
                    feature_key='q',
                   aggregation_mode='sum')

def replicate_box(r, q, box, nx=2, ny=2, nz=2):
    """Replicate the simulation box nx, ny, nz times in each direction."""
    n_atoms = r.shape[0]
    replicated_r = []
    replicated_q = []

    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = torch.tensor([ix, iy, iz], dtype=r.dtype, device=r.device) * box
                replicated_r.append(r + shift)
                replicated_q.append(q)
                
    replicated_r = torch.cat(replicated_r)
    replicated_q = torch.cat(replicated_q)
    
    new_box = torch.tensor([nx, ny, nz], dtype=r.dtype, device=r.device) * box
    return replicated_r, replicated_q, new_box

r = torch.rand(100, 3) * 10  # Random positions in a 10x10x10 box
q = torch.rand(100)  # Random charges
box = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)  # Box dimensions

# Replicate the box 2x2x2 times
replicated_r, replicated_q, new_box = replicate_box(r, q, box, nx=2, ny=2, nz=2)

ew_1 = ep.compute_potential(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box))
ew_2 = ep.compute_potential(torch.tensor(replicated_r), torch.tensor(replicated_q).unsqueeze(1), torch.tensor(new_box)) / 8
print(ew_1, ew_2)

