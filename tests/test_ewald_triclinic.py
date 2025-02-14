import torch
import sys
sys.path.append('../')
import cace
from cace.modules import EwaldPotential

ep = EwaldPotential(dl=1.5,
                    sigma=1,
                    exponent=1,
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

# set the same random seed for reproducibility
torch.manual_seed(0)
r = torch.rand(100, 3, ) * 10  # Random positions in a 10x10x10 box
q = torch.rand(100) * 2 - 1 # Random charges

#q -= torch.mean(q)
box = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)  # Box dimensions
box_3d = torch.tensor([[10.0, 0.0, 0.0],
                         [0.0, 10.0, 0.0],
                         [0.0, 0.0, 10.0]])
box_3d_2 = torch.tensor([[20.0, 0.0, 0.0],
                            [0.0, 20.0, 0.0],
                            [0.0, 0.0, 20.0]])

# Replicate the box 2x2x2 times
replicated_r, replicated_q, new_box = replicate_box(r, q, box, nx=2, ny=2, nz=2)

ew_1 = ep.compute_potential_optimized(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box))
ew_1_s = ep.compute_potential(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box))
ew_tri = ep.compute_potential_triclinic(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box_3d))
ew_2 = ep.compute_potential_optimized(replicated_r, replicated_q.unsqueeze(1), new_box) 
ew_2_s = ep.compute_potential(replicated_r, replicated_q.unsqueeze(1), new_box) 
ew_2_tri = ep.compute_potential_triclinic(replicated_r.to(dtype=torch.float32), replicated_q.to(dtype=torch.float32).unsqueeze(1), box_3d_2)
print('###cubic cell test###')
print(ew_1[0], ew_1_s[0], ew_tri[0])
print(ew_2[0], ew_2_s[0], ew_2_tri[0])
print(ew_2[0]/ew_1[0], ew_2_s[0]/ew_1_s[0], ew_2_tri[0]/ew_tri[0])


#triclinic cell test
def replicate_box_tri(r, q, box, nx=2, ny=2, nz=2):
    replicated_r = []
    replicated_q = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                shift = ix * box[0, :] + iy * box[1, :] + iz * box[2, :]
                replicated_r.append(r + shift)
                replicated_q.append(q)
    replicated_r = torch.cat(replicated_r, dim=0)
    replicated_q = torch.cat(replicated_q, dim=0)
    new_box = torch.stack([nx * box[0, :], ny * box[1, :], nz * box[2, :]], dim=0)
    return replicated_r, replicated_q, new_box


box_tric = torch.tensor([[10.0, 2.0, 1.0],
                         [0.0, 9.0, 1.5],
                         [0.0, 0.0, 10.0]])

s_rand = torch.rand(100, 3)
r_tric = torch.matmul(s_rand, box_tric)
q_tric = torch.rand(100) * 2 - 1

# Create an instance of EwaldPotential with given parameters.
ep = cace.modules.EwaldPotential(dl=2, sigma=1, exponent=1, feature_key='q', aggregation_mode='sum')

# Compute energy for the original triclinic configuration.
ew_tric = ep.compute_potential_triclinic(r_tric, q_tric.unsqueeze(1), box_tric)

# Replicate the cell 2x2x2 times.
rep_r, rep_q, new_box = replicate_box_tri(r_tric, q_tric, box_tric, nx=2, ny=2, nz=2)
# Compute energy for the replicated configuration and divide by 8 (replication factor)
ew_tric_rep = ep.compute_potential_triclinic(rep_r, rep_q.unsqueeze(1), new_box)

print('###triclinic cell test###')
print("Triclinic energy (original):", ew_tric[0])
print("Triclinic energy (replicated):", ew_tric_rep[0])
print("Ratio:", ew_tric_rep[0] / ew_tric[0])