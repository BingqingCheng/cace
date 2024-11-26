import torch
import cace
from cace.modules import EwaldPotential
import sys
torch.set_default_dtype(torch.float32)

ep = EwaldPotential(dl=1.,
                    sigma=2,
                    exponent=1,
                    feature_key='q',
                    aggregation_mode='sum',
                    remove_self_interaction=True,
                    compute_field=True)

# set the same random seed for reproducibility
torch.manual_seed(sys.argv[1])
r = torch.rand(10, 3) * 8  # Random positions in a 10x10x10 box
q = torch.rand(10) * 2 # Random charges

q -= torch.mean(q)
box = torch.tensor([30.0, 30.0, 30.0], dtype=torch.float32)  # Box dimensions

#print(q)
#exit()

# Replicate the box 2x2x2 times

ew_1, field_1 = ep.compute_potential_optimized(torch.tensor(r), torch.tensor(q).unsqueeze(1), torch.tensor(box), compute_field=True)
ew_1_s, field_1_s = ep.compute_potential_realspace(torch.tensor(r), torch.tensor(q), compute_field=True)
print(ew_1, ew_1_s)
print(ew_1.shape, ew_1_s.shape)
print(field_1, field_1_s)
print(field_1.shape, field_1_s.shape)
print(torch.sum(q.unsqueeze(1) * field_1 / 2), torch.sum(q.unsqueeze(1) * field_1_s / 2))

