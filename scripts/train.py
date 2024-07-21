import torch
import logging
import ase.io
import cace
import pickle
import os
from cace.representations import Cace
from cace.modules import PolynomialCutoff, BesselRBF, Atomwise, Forces
from cace.models.atomistic import NeuralNetworkPotential
from cace.tasks.train import TrainingTask, GetLoss
from cace.tools import Metrics, init_device, compute_average_E0s, setup_logger, get_unique_atomic_number
from cace.tools import parse_arguments

def main():
    args = parse_arguments()

    setup_logger(level='INFO', tag=args.prefix, directory='./')
    device = init_device(args.use_device)

    if args.zs is None:
        xyz = ase.io.read(args.train_path, ':')
        args.zs = get_unique_atomic_number(xyz)

# load the avge0 dict from a file if possible
    if os.path.exists('avge0.pkl'):
        with open('avge0.pkl', 'rb') as f:
            avge0 = pickle.load(f)
    else:
        # Load Dataset
        avge0 = compute_average_E0s(xyz)
        with open('avge0.pkl', 'wb') as f:
            pickle.dump(avge0, f)

    # Prepare Data Loaders
    collection = cace.tasks.get_dataset_from_xyz(
        train_path=args.train_path,
        valid_fraction=args.valid_fraction,
        data_key={'energy': args.energy_key, 'forces': args.forces_key},
        atomic_energies=avge0,
        cutoff=args.cutoff)

    train_loader = cace.tasks.load_data_loader(
        collection=collection,
        data_type='train',
        batch_size=args.batch_size)

    valid_loader = cace.tasks.load_data_loader(
        collection=collection,
        data_type='valid',
        batch_size=args.valid_batch_size)

    # Configure CACE Representation
    cutoff_fn = PolynomialCutoff(cutoff=args.cutoff, p=args.cutoff_fn_p)
    radial_basis = BesselRBF(cutoff=args.cutoff, n_rbf=args.n_rbf, trainable=args.trainable_rbf)
    cace_representation = Cace(
        zs=args.zs, n_atom_basis=args.n_atom_basis, embed_receiver_nodes=args.embed_receiver_nodes,
        cutoff=args.cutoff, cutoff_fn=cutoff_fn, radial_basis=radial_basis,
        n_radial_basis=args.n_radial_basis, max_l=args.max_l, max_nu=args.max_nu,
        device=device, num_message_passing=args.num_message_passing)

    # Configure Atomwise Module
    atomwise = Atomwise(
        n_layers=args.atomwise_layers, n_hidden=args.atomwise_hidden, residual=args.atomwise_residual,
        use_batchnorm=args.atomwise_batchnorm, add_linear_nn=args.atomwise_linear_nn,
        output_key='CACE_energy')

    # Configure Forces Module
    forces = Forces(energy_key='CACE_energy', forces_key='CACE_forces')

    # Assemble Neural Network Potential
    cace_nnp = NeuralNetworkPotential(representation=cace_representation, output_modules=[atomwise, forces]).to(device)

    # Phase 1 Training Configuration
    optimizer_args = {'lr': args.lr}
    scheduler_args = {'mode': 'min', 'factor': args.scheduler_factor, 'patience': args.scheduler_patience}
    energy_loss = GetLoss(
        target_name='energy',
        predict_name='CACE_energy',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=args.energy_loss_weight)
    force_loss = GetLoss(
        target_name='forces', 
        predict_name='CACE_forces', 
        loss_fn=torch.nn.MSELoss(), 
        loss_weight=args.force_loss_weight)


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

    for _ in range(args.num_restart): 
        # Initialize and Fit Training Task for Phase 1
        task = TrainingTask(
            model=cace_nnp, losses=[energy_loss, force_loss], metrics=[e_metric, f_metric],
            device=device, optimizer_args=optimizer_args, scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
            scheduler_args=scheduler_args, max_grad_norm=args.max_grad_norm, ema=args.ema,
            ema_start=args.ema_start, warmup_steps=args.warmup_steps)

        task.fit(train_loader, valid_loader, epochs=int(args.epochs/args.num_restart), print_stride=0)
    task.save_model(args.prefix+'_phase_1.pth')

    # Phase 2 Training Adjustment
    energy_loss_2 = GetLoss('energy', 'CACE_energy', torch.nn.MSELoss(), args.second_phase_energy_loss_weight)
    task.update_loss([energy_loss_2, force_loss])

    # Fit Training Task for Phase 2
    task.fit(train_loader, valid_loader, epochs=args.second_phase_epochs)
    task.save_model(args.prefix+'_phase_2.pth')

if __name__ == '__main__':
    main()

