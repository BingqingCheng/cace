###########################################################################################
# Utilities
# modified from MACE
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Optional, Tuple

import torch

__all__ = ["get_outputs", "get_edge_vectors_and_lengths", "get_edge_node_type", "get_symmetric_displacement"]

def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = False
) -> torch.Tensor:
    # check the dimension of the energy tensor
    if len(energy.shape) == 1:
        grad_outputs= torch.jit.annotate(Optional[List[Optional[torch.Tensor]]], [torch.ones_like(energy)])
        gradient = torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=[positions],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]  # [n_nodes, 3]
        if gradient is None:
            gradient = torch.zeros_like(positions) #added
    else:
        num_energy = energy.shape[1]
        gradient_list = []
        for i in range(num_energy):
            # **Explicitly annotate grad_outputs inside the loop**
            grad_outputs = torch.jit.annotate(Optional[List[Optional[torch.Tensor]]], [torch.ones_like(energy[:, i])])
            grad = torch.autograd.grad(
                outputs=[energy[:, i]], # [n_graphs, ]
                inputs=[positions], # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < num_energy - 1)), # Make sure the graph is not destroyed during training
                create_graph=(training or (i < num_energy - 1)), # Create graph for second derivative
                allow_unused=True, # For complete dissociation turn to true
            )[0]
            if grad is None:
                grad = torch.zeros_like(positions)
            gradient_list.append(grad)
        gradient = torch.stack(gradient_list, dim=2)  # [n_nodes, 3, num_energy]  

    return -1 * gradient


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = False,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # if displacement is None:
    #     displacement = torch.zeros_like(positions)
    # if cell is None:
    #     cell = torch.zeros((positions.shape[0], 3, 3), dtype=positions.dtype, device=positions.device)
    if len(energy.shape) == 1:
        grad_outputs = torch.jit.annotate(Optional[List[Optional[torch.Tensor]]], [torch.ones_like(energy)]) 
        inputs = [positions, displacement]
        grads = torch.autograd.grad(
            outputs=[energy],  # [n_graphs, ]
            inputs=inputs,  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=training,  # Make sure the graph is not destroyed during training
            create_graph=training,  # Create graph for second derivative
            allow_unused=True,
        )
        gradient = grads[0]
        virials = grads[1]

        if gradient is None:
            gradient = torch.zeros_like(positions)
        if virials is None:
            virials = torch.zeros_like(displacement)
        if compute_stress:
            cell = cell.view(-1, 3, 3)
            volume = torch.einsum(
                "zi,zi->z",
                cell[:, 0, :],
                torch.cross(cell[:, 1, :], cell[:, 2, :], dim=1),
            ).unsqueeze(-1)
            stress = virials / volume.view(-1, 1, 1)
        else:
            stress = torch.zeros_like(cell)
    else:
        raise NotImplementedError("Multiple energy outputs not supported for forces and virials")

    return -1 * gradient, -1 * virials, stress

def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement

def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    if (compute_virials or compute_stress) and displacement is not None:
        # forces come for free
        forces, virials, stress = compute_forces_virials(
            energy=energy,
            positions=positions,
            displacement=displacement,
            cell=cell,
            compute_stress=compute_stress,
            training=training,
        )
    elif compute_force:
        forces, virials, stress = (
            compute_forces(energy=energy, positions=positions, training=training),
            None,
            None,
        )
    else:
        forces, virials, stress = (None, None, None)
    return forces, virials, stress

def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    if normalize:
        vectors_normed = vectors / (lengths + eps)
        return vectors_normed, lengths

    return vectors, lengths

def get_edge_node_type(
    edge_index: torch.Tensor,  # [2, n_edges]
    node_type: torch.Tensor,  # [n_nodes, n_dims]
    node_type_2: Optional[torch.Tensor]=None,  # [n_nodes, n_dims]
) -> Tuple[torch.Tensor, torch.Tensor]:
    if node_type_2 is None:
        node_type_2 = node_type.clone()

    edge_type = torch.zeros([edge_index.shape[1], 2, node_type.shape[1]], 
                           dtype=node_type.dtype, device=node_type.device)
    sender_type = node_type[edge_index[0]]
    receiver_type = node_type_2[edge_index[1]]
    return sender_type, receiver_type  # [n_edges, n_dims]

