import torch
from cace.representations import Cace
from cace import data
from cace.modules import BesselRBF
from cace.modules import PolynomialCutoff

#Model
cutoff = 4.0
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

representation = Cace(
    zs=[1,8],
    n_atom_basis=4,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_l_out=2,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=["M", "Ar", "Bchi"],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    timeit=False,
)

from cace.modules.forces import DirectForces
model = DirectForces()

from ase import Atoms
from ase.io import read

root_xyz = f"test_datasets/water_dimer.xyz"
atom = read(root_xyz)
root_xyz_r = f"test_datasets/water_dimer_r.xyz"
atom_rotated = read(root_xyz_r)

atomic_data = data.AtomicData.from_atoms(atom, cutoff=cutoff)
atomic_data_r = data.AtomicData.from_atoms(atom_rotated, cutoff=cutoff)

out = model(representation(atomic_data))
rout = model(representation(atomic_data_r))

#Feature equivariance
a1 = out["node_feats_l"][1]   # N x C x 3
b1 = rout["node_feats_l"][1][:,:,[1,0,2]]   # N x C x 3
assert torch.allclose(a1, b1, atol=1e-6), "Level l=1 equivariance test failed: outputs do not match under permutation."
print("Level l=1 equivariance test passed.")

a2 = out["node_feats_l"][2]   # N x C x 3
b2 = rout["node_feats_l"][2][:,:,[1,0,2],:][:,:,:,[1,0,2]]   # N x C x 3
assert torch.allclose(a2, b2, atol=1e-6), "Level l=2 equivariance test failed: outputs do not match under permutation."
print("Level l=2 equivariance test passed.")

a = out["forces"]
b = rout["forces"][:, [1, 0, 2]]
assert torch.allclose(a, b, atol=1e-6), "Equivariance test failed: Forces are not invariant under permutation."
print("Equivariance test passed: Forces are invariant under permutation.")
