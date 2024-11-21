# Recommended metatensor version
# metatensor                0.2.0
# metatensor-core           0.1.10
# metatensor-learn          0.2.0
# metatensor-operations     0.2.4
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
    def __init__(self, model_path: str, atomic_energies: Dict[int, float] = None, device: str = 'cpu'):
        super().__init__()
        # Load the pre-trained TorchScript model
        self.model = torch.jit.load(model_path, map_location=device)
        self.atomic_energies = atomic_energies

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        return [
            NeighborListOptions(
                cutoff=5.5,       # cutoff radius
                full_list=True,   # full neighbor list
                strict=True,      # strict neighbor list
            ),            
            NeighborListOptions(
                cutoff=5.5,       # cutoff radius
                full_list=False,   # half neighbor list
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

        # Iterate over the systems
        for system_index, system in enumerate(systems):
            # Extract tensor data from the System object
            positions = system.positions  # Tensor [n_nodes, 3]
            atomic_numbers = system.types  # Tensor [n_nodes]
            cell = system.cell  # Tensor [3, 3]
            # Extract data from the neighbor list
            neighbors: TensorBlock = system.get_neighbor_list(neighbor_list_options)
            edge_index = torch.stack([ neighbors.samples.values[:, 0].to(torch.long),  # first_atom
                                        neighbors.samples.values[:, 1].to(torch.long)])   # second_atom
            unit_shifts = neighbors.samples.values[:, 2:5]
            shifts = torch.matmul(unit_shifts.to(cell.dtype), cell)
            # Batch tensor (all zeros since it's a single system)
            batch = torch.zeros(atomic_numbers.shape[0], dtype=torch.long)

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

            if self.atomic_energies:
                e0 = torch.scalar_tensor(0.0, dtype=torch.float64)
                for i in range(len(atomic_numbers)):
                    Z = int(atomic_numbers[i])
                    e0 += self.atomic_energies.get(Z, 0.0)
            else:
                e0 = torch.scalar_tensor(0.0)
            
            # Extract energy
            energy: Tensor = prediction['CACE_energy'] + e0  # Modify to match the model's output key
            # make sure no forces is here to solve doulbe backward error
            if 'CACE_forces' in prediction and prediction['CACE_forces'] is not None:
                raise ValueError(f"{prediction['CACE_forces'][0]}")

            # Append energy and system index
            energies_list.append(energy.view(1, 1))
            samples_list.append(system_index)

        # Stack the energy tensors
        energies: Tensor = torch.cat(energies_list, dim=0)  # Shape: (num_systems, 1)
        # Create sample labels
        samples_values = torch.tensor(samples_list, dtype=torch.int64).unsqueeze(1)
        samples = Labels(
            names=["system"],
            values=samples_values
        )
        # Create property labels (including CACE_energy)
        properties = Labels(
            names=["CACE_energy"],  # Add energy name
            values=torch.tensor([[0]], dtype=torch.int64)  # Single property index
        )
        # Create a TensorBlock
        energy_block = TensorBlock(
            values=energies,
            samples=samples,
            components=[],  # Define if needed
            properties=properties
        )
        # Create key labels
        keys = Labels(
            names=["_"],  # Use a key name indicated by metatensor document
            values=torch.tensor([[0]], dtype=torch.int64)  # [1, 1]
        )
        # Create a TensorMap
        energy_tensormap = TensorMap(
            keys=keys,
            blocks=[energy_block]
        )
        # Add to results
        results['energy'] = energy_tensormap

        return results