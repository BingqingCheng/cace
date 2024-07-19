import argparse

__all__ = ['parse_arguments']

def parse_arguments():
    parser = argparse.ArgumentParser(description='ML potential training configuration')

    parser.add_argument('--prefix', type=str, default='CACE_NNP', help='Prefix for the model name')

    # Dataset and Training Configuration
    parser.add_argument('--train_path', type=str, help='Path to the training dataset', required=True)
    parser.add_argument('--valid_path', type=str, default=None, help='Path to the training dataset')
    parser.add_argument('--valid_fraction', type=float, default=0.1, help='Fraction of data to use for validation')
    parser.add_argument('--energy_key', type=str, default='energy', help='Key for energy in the dataset')
    parser.add_argument('--forces_key', type=str, default='forces', help='Key for forces in the dataset')
    parser.add_argument('--cutoff', type=float, default=4.0, help='Cutoff radius for interactions')
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size for training')
    parser.add_argument('--valid_batch_size', type=int, default=20, help='Batch size for validation')
    parser.add_argument('--use_device', type=str, default='cpu', help='Device to use for training')

    # Radial Basis Function (RBF) Configuration
    parser.add_argument('--n_rbf', type=int, default=6, help='Number of RBFs')
    parser.add_argument('--trainable_rbf', action='store_true', help='Whether the RBF parameters are trainable')

    # Cutoff Function Configuration
    parser.add_argument('--cutoff_fn', type=str, default='PolynomialCutoff', help='Type of cutoff function')
    parser.add_argument('--cutoff_fn_p', type=int, default=5, help='Polynomial degree for the PolynomialCutoff function')

    # Representation Configuration
    parser.add_argument('--zs', type=int, nargs='+', default=None, help='Atomic numbers considered in the model')
    parser.add_argument('--n_atom_basis', type=int, default=3, help='Number of atom basis functions')
    parser.add_argument('--n_radial_basis', type=int, default=8, help='Number of radial basis functions')
    parser.add_argument('--max_l', type=int, default=3, help='Maximum angular momentum quantum number')
    parser.add_argument('--max_nu', type=int, default=3, help='Maximum radial quantum number')
    parser.add_argument('--num_message_passing', type=int, default=1, help='Number of message passing steps')
    parser.add_argument('--embed_receiver_nodes', action='store_true', help='Whether to embed receiver nodes')

    # Atomwise Module Configuration
    parser.add_argument('--atomwise_layers', type=int, default=3, help='Number of layers in the atomwise module')
    parser.add_argument('--atomwise_hidden', type=int, nargs='+', default=[32, 16], help='Hidden units in each layer of the atomwise module')
    parser.add_argument('--atomwise_residual', action='store_false', help='Use residual connections in the atomwise module')
    parser.add_argument('--atomwise_batchnorm', action='store_false', help='Use batch normalization in the atomwise module')
    parser.add_argument('--atomwise_linear_nn', action='store_true', help='Add a linear neural network layer in the atomwise module')

    # Training Procedure Configuration
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--scheduler_factor', type=float, default=0.8, help='Factor by which the learning rate is reduced')
    parser.add_argument('--scheduler_patience', type=int, default=10, help='Patience for the learning rate scheduler')
    parser.add_argument('--max_grad_norm', type=float, default=10, help='Max gradient norm for gradient clipping')
    parser.add_argument('--ema', action='store_true', help='Use exponential moving average of model parameters')
    parser.add_argument('--ema_start', type=int, default=10, help='Start using EMA after this many steps')
    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps for the optimizer')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs for the first phase of training')
    parser.add_argument('--second_phase_epochs', type=int, default=100, help='Number of epochs for the second phase of training')
    parser.add_argument('--energy_loss_weight', type=float, default=1.0, help='Weight for the energy loss in phase 1')
    parser.add_argument('--force_loss_weight', type=float, default=1000.0, help='Weight for the force loss in both phases')
    parser.add_argument('--num_restart', type=int, default=5, help='Number of restarts for the training during phase 1')
    parser.add_argument('--second_phase_energy_loss_weight', type=float, default=1000.0, 
                        help='Weight for the energy loss in phase 2')

    return parser.parse_args()
