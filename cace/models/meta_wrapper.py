# Recommended metatensor version
# metatensor                0.2.0
# metatensor-core           0.1.10
# metatensor-learn          0.3.0
# metatensor-operations     0.3.0
# metatensor-torch          0.6.1
import torch
import torch.nn as nn
from typing import List, Dict, Optional
from metatensor.torch import Labels, TensorBlock, TensorMap
from metatensor.torch.atomistic import System, ModelOutput
from torch import Tensor
from metatensor.torch.atomistic import NeighborListOptions

__all__ = ["MetatensorWrapper"]


# Definition of the MetatensorWrapper class
class MetatensorWrapper(nn.Module):
    def __init__(self, 
        model_path: str, atomic_energies: Dict[int, float] = None, 
        # device: str = 'cpu', 
        energy_key: str='CACE_energy', forces_key: str='CACE_forces',
        cutoff: float = 5.5
        ):
        super().__init__()
        # Load the pre-trained TorchScript model
        self.model = torch.jit.load(model_path)
        self.atomic_energies = atomic_energies
        self.energy_key = energy_key
        self.forces_key = forces_key
        self.cutoff = cutoff

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=self.cutoff,       # cutoff radius
                full_list=True,   # full neighbor list
                strict=True,      # strict neighbor list
            )]
    
    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None
    ) -> Dict[str, TensorMap]:
        """
        Forward method designed to match the expected signature of Metatensor.

        Args:
            systems (List[System]): List of systems.
            outputs (Dict[str, ModelOutput]): Outputs to predict.
            selected_atoms (Optional[Labels]): Information about selected atoms.

        Returns:
            Dict[str, TensorMap]: Prediction results.
        """
        # Initialize the results dictionary
        results: Dict[str, TensorMap] = {}

        # Check if 'energy' output is requested
        if 'energy' not in outputs:
            raise ValueError("This model only supports 'energy' output.")

        # Initialize lists to collect energies and samples
        energies_list: List[Tensor] = []
        samples_list: List[int] = []

        #get_neighbor_list_option
        neighbor_list_options = self.requested_neighbor_lists()[0]

        #set the device
        device = systems[0].positions.device

        # Iterate over the systems
        for system_index, system in enumerate(systems):

            # Extract tensor data from the System object
            positions = system.positions.to(device)  # Tensor [n_nodes, 3]
            atomic_numbers = system.types.to(device)  # Tensor [n_nodes]
            cell = system.cell.to(device)  # Tensor [3, 3]

            # Extract data from the neighbor list
            neighbors: TensorBlock = system.get_neighbor_list(neighbor_list_options)
            edge_index = torch.stack([ 
                neighbors.samples.values[:, 0].to(torch.long).to(device),  # first_atom
                neighbors.samples.values[:, 1].to(torch.long).to(device)   # second_atom
                ])   
            unit_shifts = neighbors.samples.values[:, 2:5].to(device) 
            shifts = torch.matmul(unit_shifts.to(cell.dtype), cell)
            
            # Batch tensor (all zeros since it's a single system)
            batch = torch.zeros(atomic_numbers.shape[0], dtype=torch.long, device=device)

            # Prepare the data dictionary
            data: Dict[str, Tensor] = {
                'positions': positions,
                'atomic_numbers': atomic_numbers,
                'batch': batch,
                'edge_index': edge_index,
                'shifts': shifts,
                'unit_shifts': unit_shifts,
                'cell': cell,
            }

            # Run the model (torch.no_grad() is removed to allow gradient tracking)
            prediction = self.model(data)
            dtype= prediction[self.energy_key].dtype

            if self.atomic_energies:
                e0 = torch.scalar_tensor(0.0, dtype=torch.float64, device=device)
                for i in range(len(atomic_numbers)):
                    Z = int(atomic_numbers[i])
                    e0 += torch.scalar_tensor(self.atomic_energies.get(Z, 0.0), dtype=dtype, device=device)
            else:
                e0 = torch.scalar_tensor(0.0, dtype=dtype, device=device)
            
            # Extract energy
            energy: Tensor = prediction[self.energy_key] + e0 
            # make sure no forces is here to solve doulbe backward error
            if self.forces_key in prediction and prediction[self.forces_key] is not None:
                raise ValueError(f"{prediction[self.forces_key][0]}")

            # Append energy and system index
            energies_list.append(energy.view(1, 1))
            samples_list.append(system_index)

        # Stack the energy tensors
        energies: Tensor = torch.cat(energies_list, dim=0)  # Shape: (num_systems, 1)

        # Create sample labels
        samples_values = torch.tensor(samples_list, dtype=torch.int32, device=device).unsqueeze(1)
        samples = Labels(
            names=["system"],
            values=samples_values
        )
        # Create property labels (including CACE_energy)
        properties = Labels(
            names=["energy"],  # the energy must have a single property dimension named "energy"
            values=torch.tensor([[0]], dtype=torch.int32, device=device)  # with a single entry set to 0.
        )
        # Create a TensorBlock
        energy_block = TensorBlock(
            values=energies,
            samples=samples,
            components=[],  # the energy must not have any components
            properties=properties
        )
        # Create key labels
        keys = Labels(
            names=["_"],  # the energy keys must have a single dimension named "_"
            values=torch.tensor([[0]], dtype=torch.int32, device=device)  # , with a single entry set to 0
        )
        # Create a TensorMap
        energy_tensormap = TensorMap(
            keys=keys,
            blocks=[energy_block]
        )
        # Add to results
        results['energy'] = energy_tensormap

        return results