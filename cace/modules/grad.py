from typing import Dict
import torch
from torch import nn

__all__ = ['Grad']

class Grad(nn.Module):
    """
    a wrapper for the gradient calculation

    """

    def __init__(
        self,
        y_key: str,
        x_key: str,
        output_key: str = 'gradient',
    ):
        super().__init__()
        self.y_key = y_key
        self.x_key = x_key
        self.output_key = output_key
        self.required_derivatives = [self.x_key]
        self.model_outputs = [self.output_key]

    def forward(self, data: Dict[str, torch.Tensor], training: bool = True, output_index: int = None) -> Dict[str, torch.Tensor]:
        y = data[self.y_key]
        x = data[self.x_key]

        if y.is_complex():
            get_imag = True
        else:
            get_imag = False

        if len(y.shape) == 1:
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y)]
            gradient_real = torch.autograd.grad(
                outputs=[y],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or get_imag),  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
            )[0]  # [n_nodes, 3]

            if get_imag:
                gradient_imag = torch.autograd.grad(
                    outputs=[y/1j],  # [n_graphs, ]
                    inputs=[x],  # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=training,  # Make sure the graph is not destroyed during training
                    create_graph=training,  # Create graph for second derivative
                    allow_unused=True,  # For complete dissociation turn to true
                )[0]  # [n_nodes, 3]
            else:
                gradient_imag = 0.0
        else:
            dim_y = y.shape[1] 
            grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(y[:,0])]
            gradient_real = torch.stack([
                torch.autograd.grad(
                outputs=[y[:,i]],  # [n_graphs, ]
                inputs=[x],  # [n_nodes, 3]
                grad_outputs=grad_outputs,
                retain_graph=(training or (i < dim_y - 1) or get_imag),  # Make sure the graph is not destroyed during training
                create_graph=training,  # Create graph for second derivative
                allow_unused=True,  # For complete dissociation turn to true
                )[0] for i in range(dim_y)
               ], axis=2)  # [n_nodes, 3, num_energy]
            # if y is complex, we need to calculate the imaginary part
            if get_imag:
                gradient_imag = torch.stack([
                    torch.autograd.grad(
                    outputs=[y[:,i]/1j],  # [n_graphs, ]
                    inputs=[x],  # [n_nodes, 3]
                    grad_outputs=grad_outputs,
                    retain_graph=(training or (i < dim_y - 1)),  # Make sure the graph is not destroyed during training
                    create_graph=training,  # Create graph for second derivative
                    allow_unused=True,  # For complete dissociation turn to true
                    )[0] for i in range(dim_y)
                   ], axis=2)  # [n_nodes, 3, num_energy]
        if get_imag:
            data[self.output_key] = gradient_real + 1j * gradient_imag
        else:
            data[self.output_key] = gradient_real
        
        return data 

    def __repr__(self):
        return (
            f"{self.__class__.__name__} (function={self.y_key}, variable={self.x_key},) "
            )
