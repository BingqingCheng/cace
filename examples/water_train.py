#!/usr/bin/env python
# coding: utf-8

import sys
sys.path.append('../cace/')

import numpy as np
#import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import logging

import cace
from cace.representations import Cace
from cace.modules import CosineCutoff, MollifierCutoff, PolynomialCutoff
from cace.modules import BesselRBF, GaussianRBF, GaussianRBFCentered

from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask

torch.set_default_dtype(torch.float32)

cace.tools.setup_logger(level='INFO')

logging.info("reading data")
collection = cace.tasks.get_dataset_from_xyz(train_path='../../datasets/water/water.xyz',
                                 valid_fraction=0.1,
                                 seed=1,
                                 energy_key='energy',
                                 forces_key='force',
                                 atomic_energies={1: -187.6043857100553, 8: -93.80219285502734} # avg
                                 )
cutoff = 5.5
batch_size = 2

train_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='train',
                              batch_size=batch_size,
                              cutoff=cutoff)

valid_loader = cace.tasks.load_data_loader(collection=collection,
                              data_type='valid',
                              batch_size=4,
                              cutoff=cutoff)

use_device = 'cuda'
device = cace.tools.init_device(use_device)
logging.info(f"device: {use_device}")


logging.info("building CACE representation")
radial_basis = BesselRBF(cutoff=cutoff, n_rbf=6, trainable=True)
#cutoff_fn = CosineCutoff(cutoff=cutoff)
cutoff_fn = PolynomialCutoff(cutoff=cutoff)

cace_representation = Cace(
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
    device=device,
    timeit=False
           )

cace_representation.to(device)
logging.info(f"Representation: {cace_representation}")

atomwise = cace.modules.atomwise.Atomwise(n_layers=3,
                                         output_key='CACE_energy',
                                         n_hidden=[32,16],
                                         use_batchnorm=False,
                                         add_linear_nn=True)


forces = cace.modules.forces.Forces(energy_key='CACE_energy',
                                    forces_key='CACE_forces')

logging.info("building CACE NNP")
cace_nnp = NeuralNetworkPotential(
    input_modules=None,
    representation=cace_representation,
    output_modules=[atomwise, forces]
)

#trainable_params = sum(p.numel() for p in cace_nnp.parameters() if p.requires_grad)
#logging.info(f"Number of trainable parameters: {trainable_params}")

cace_nnp.to(device)


logging.info(f"First train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.1
)

force_loss = cace.tasks.GetLoss(
    target_name='forces',
    predict_name='CACE_forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=1000
)

from cace.tools import Metrics

e_metric = Metrics(
    target_name='energy',
    predict_name='CACE_energy',
    name='e/atom',
    per_atom=True
)

f_metric = Metrics(
    target_name='forces',
    predict_name='CACE_forces',
    name='f'
)

# Example usage
logging.info("creating training task")

optimizer_args = {'lr': 1e-2}  
scheduler_args = {'mode': 'min', 'factor': 0.8, 'patience': 10}

for i in range(5):
    task = TrainingTask(
        model=cace_nnp,
        losses=[energy_loss, force_loss],
        metrics=[e_metric, f_metric],
        device=device,
        optimizer_args=optimizer_args,
        #scheduler_cls=torch.optim.lr_scheduler.StepLR,
        scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
        scheduler_args=scheduler_args,
        max_grad_norm=10,
        ema=True,
        ema_start=10,
        warmup_steps=5,
    )

    logging.info("training")
    task.fit(train_loader, valid_loader, epochs=40, screen_nan=False)

task.save_model('water-model.pth')
cace_nnp.to(device)

logging.info(f"Second train loop:")
energy_loss = cace.tasks.GetLoss(
    target_name='energy',
    predict_name='CACE_energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=100
)

task.update_loss([energy_loss, force_loss])
logging.info("training")
task.fit(train_loader, valid_loader, epochs=100, screen_nan=False)


task.save_model('water-model-2.pth')
cace_nnp.to(device)

logging.info(f"Finished")





