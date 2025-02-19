import torch, math
import sys
sys.path.append('../')
import cace
from cace.modules import Polarization

#generate random data
torch.random.manual_seed(0)
r_now = torch.randn(192,3, dtype=torch.float64)
q_now = torch.randn(192,1, dtype=torch.float64)

#box for orignal function
box_now = torch.tensor([10.,10.,10.])

#calculate polarization using original function
factor_o = box_now / (1j * 2.* torch.pi)
phase_o = torch.exp(1j * 2.* torch.pi * r_now / box_now)
polarization_o = torch.sum(q_now * phase_o, dim=(0)) * factor_o
print("Original polarization:", polarization_o)

#calculate polarization using new function
pol_class = Polarization(pbc=True)
box_ortho = torch.tensor([[10.0, 0.0, 0.0],
                    [0.0, 10.0, 0.0],
                    [0.0, 0.0, 10.0]], dtype=torch.float64)
pol_ortho, phase_ortho = pol_class.compute_pol_pbc(r_now, q_now, box_ortho)
print("Modified polarization:", pol_ortho)
print(torch.allclose(polarization_o, pol_ortho))


# Check that the polarization is equivalent to the rotated polarization in the triclinic cell

#Rotation matrix around z-axis, 45 degrees
theta = math.pi/4
R = torch.tensor([[math.cos(theta), -math.sin(theta), 0],
                  [math.sin(theta),  math.cos(theta), 0],
                  [0,               0,              1]], dtype=torch.float64)

# Triclinic cell
box_tric = torch.matmul(R, box_ortho)
r_tric = torch.matmul(r_now, R)

pol_tric, phase_tric = pol_class.compute_pol_pbc(r_tric, q_now, box_tric)
pol_expected = torch.matmul(R.to(torch.complex128), pol_ortho.unsqueeze(1)).squeeze()

print("Triclinic polarization:", pol_tric)
print("Expected triclinic polarization from rotation:", pol_expected)
print(torch.allclose(pol_tric, pol_expected))