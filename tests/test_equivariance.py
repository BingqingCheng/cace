import torch
from cace.representations import Cace
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

from cace.data.xyzdata import XYZData
root_xyz = f"test_datasets/water_dimer.xyz"
data = XYZData(root_xyz,batch_size=4,cutoff=cutoff) 

from cace.modules.forces import DirectForces
model = DirectForces()
model.cuda()

representation.cuda()
for batch in data.val_dataloader():
    batch.cuda()
    out = representation(batch)
    break
batch["node_feats_l"] = out["node_feats_l"]
forces = model(batch)["forces"]

root_xyz = f"test_datasets/water_dimer_r.xyz"
rdata = XYZData(root_xyz,batch_size=4,cutoff=cutoff) 
for rbatch in rdata.val_dataloader():
    rbatch.cuda()
    rout = representation(rbatch)
    break
rbatch["node_feats_l"] = rout["node_feats_l"]
rforces = model(rbatch)["forces"]

#Feature equivariance
a1 = out["node_feats_l"][1]   # N x C x 3
b1 = rout["node_feats_l"][1][:,:,[1,0,2]]   # N x C x 3
assert torch.allclose(a1, b1, atol=1e-6), "Level l=1 equivariance test failed: outputs do not match under permutation."
print("Level l=1 equivariance test passed.")

a2 = out["node_feats_l"][2]   # N x C x 3
b2 = rout["node_feats_l"][2][:,:,[1,0,2],:][:,:,:,[1,0,2]]   # N x C x 3
assert torch.allclose(a2, b2, atol=1e-6), "Level l=2 equivariance test failed: outputs do not match under permutation."
print("Level l=2 equivariance test passed.")

a = forces
b = rforces[:, [1, 0, 2]]
assert torch.allclose(a, b, atol=1e-6), "Equivariance test failed: Forces are not invariant under permutation."
print("Equivariance test passed: Forces are invariant under permutation.")