import torch
import torch.nn as nn
import numpy as np

class SharedRadialLinearTransform(nn.Module):
    def __init__(self, max_l: int, radial_dim: int):
        super().__init__()
        self.max_l = max_l
        self.radial_dim = radial_dim
        self.angular_dim_groups = self._init_angular_dim_groups(max_l)
        self.weights = self._initialize_weights(radial_dim)

    def _initialize_weights(self, radial_dim: int):
        return nn.ParameterList([
            nn.Parameter(torch.eye(radial_dim)) for _ in self.angular_dim_groups 
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        n_nodes, radial_dim, angular_dim, embedding_dim = x.shape

        output = torch.zeros_like(x)
        """ this is the original code
        for group_idx, group in enumerate(self.angular_dim_groups):
            weight = self.weights[group_idx]
            # Apply the transformation for each group
            for ang_dim in group:
                transformed = torch.matmul(x[:, :, ang_dim, :], weight)
                output[:, :, ang_dim, :] = transformed
        """

        # this batch the transformation for each group, slower than the original code on cpus, need to test on gpus
        for group_idx, group in enumerate(self.angular_dim_groups):
            weight = self.weights[group_idx]

            # Gather all angular dimensions for the current group
            group_x = x[:, :, group, :]  # Shape: [n_nodes, radial_dim, len(group), embedding_dim]

            # Reshape for batch operation: [n_nodes * len(group), embedding_dim, radial_dim]
            group_x_reshaped = group_x.permute(0, 2, 3, 1).reshape(-1, embedding_dim, radial_dim)
            
            # Apply the transformation for the entire group at once
            transformed_group = torch.matmul(group_x_reshaped, weight)

            # Reshape back to [n_nodes, len(group), embedding_dim, radial_dim]
            transformed_group = transformed_group.reshape(n_nodes, len(group), embedding_dim, radial_dim)
            transformed_group = transformed_group.permute(0, 3, 1, 2)  # Back to [n_nodes, radial_dim, len(group), embedding_dim]

            # Assign to the output tensor for each angular dimension
            for idx, ang_dim in enumerate(group):
                output[:, :, ang_dim, :] = transformed_group[:, :, idx, :]

        return output

    def _compute_length_lxlylz(self, l):
        return int((l+1)*(l+2)/2)

    def _init_angular_dim_groups(self, max_l):
        angular_dim_groups = []
        l_now = 0
        for l in range(max_l+1):
            l_list_atl = [l_idx + l_now for l_idx in np.arange(self._compute_length_lxlylz(l))]
            angular_dim_groups.append(l_list_atl)
            l_now += len(l_list_atl)
        return angular_dim_groups

"""
n_nodes = 10
radial_dim = 5
embedding_dim = 4
A = torch.randn(n_nodes, radial_dim, 20, embedding_dim, device=device)
transform_layer =  cace.modules.SharedRadialLinearTransform(max_l=3, radial_dim=5)
A_transformed = transform_layer(A)

# Check if the transformation preserves the old tensor
print(torch.allclose(A, A_transformed))  # Should output True for the initial run
"""
