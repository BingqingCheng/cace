import os
import glob
import torch
from cace.cace.tasks import LightningData, LightningTrainingTask

on_cluster = False
if 'SLURM_JOB_CPUS_PER_NODE' in os.environ.keys():
    on_cluster = True
if on_cluster:
    root = ".../data/water.xyz" #edit paths here
else:
    root = ".../cacefit/fit-water/water.xyz" #or here

logs_name = "water_test"
cutoff = 5.5
avge0 = {1: -187.6043857100553, 8: -93.80219285502734}
batch_size = 4
data = LightningData(root,batch_size=batch_size,cutoff=cutoff,atomic_energies=avge0)

from cace.cace.representations import Cace
from cace.cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered
from cace.cace.modules import PolynomialCutoff

#Model
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

representation = Cace(
    zs=[1,8],
    n_atom_basis=3,
    embed_receiver_nodes=True,
    cutoff=cutoff,
    cutoff_fn=cutoff_fn,
    radial_basis=radial_basis,
    n_radial_basis=12,
    max_l=3,
    max_nu=3,
    num_message_passing=1,
    type_message_passing=['Bchi'],
    args_message_passing={'Bchi': {'shared_channels': False, 'shared_l': False}},
    avg_num_neighbors=1,
    timeit=False
)

for batch in data.train_dataloader():
    exdatabatch = batch
    break

from cace.cace.models import NeuralNetworkPotential
from cace.cace.modules.atomwise import Atomwise
from cace.cace.modules.forces import Forces

atomwise = Atomwise(n_layers=3,
                    output_key="pred_energy",
                    n_hidden=[32,16],
                    n_out=1,
                    use_batchnorm=False,
                    add_linear_nn=True)

forces = Forces(energy_key="pred_energy",
                forces_key="pred_force")

model = NeuralNetworkPotential(
    input_modules=None,
    representation=representation,
    output_modules=[atomwise,forces]
)

from cace.cace.tasks.lightning import default_losses, default_metrics
losses = default_losses()
metrics = default_metrics()

#Init lazy layers
for batch in data.train_dataloader():
    exdatabatch = batch
    break
model(exdatabatch)

#Check for checkpoint and restart if found:
chkpt = None
dev_run = False
if os.path.isdir(f"lightning_logs/{logs_name}"):
    latest_version = None
    num = 0
    while os.path.isdir(f"lightning_logs/{logs_name}/version_{num}"):
        latest_version = f"lightning_logs/{logs_name}/version_{num}"
        num += 1
    if latest_version:
        chkpt = glob.glob(f"{latest_version}/checkpoints/*.ckpt")[0]
if chkpt:
    print("Checkpoint found!",chkpt)
    print("Restarting...")
    dev_run = False

progress_bar = True
if on_cluster:
    torch.set_float32_matmul_precision('medium')
    progress_bar = False
task = LightningTrainingTask(model,losses=losses,metrics=metrics,log_rmse=True,
                             scheduler_args={'mode': 'min', 'factor': 0.8, 'patience': 10},
                             optimizer_args={'lr': 0.01},
                            )
task.fit(data,dev_run=dev_run,max_epochs=500,chkpt=chkpt,name=logs_name, progress_bar=progress_bar)
