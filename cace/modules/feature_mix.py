import torch
import torch.nn as nn
from typing import Dict

__all__ = ['FeatureAdd', 'FeatureInteract']

class FeatureAdd(nn.Module):
    """
    A class for adding together different features of the data.
    """
    def __init__(self,
                 feature_keys: list,
                 output_key: str):
        super().__init__()
        self.feature_keys = feature_keys
        self.output_key = output_key
        self.model_outputs = [output_key]

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        feature_shape = data[self.feature_keys[0]].shape
        result = torch.zeros_like(data[self.feature_keys[0]])
        for feature_key in self.feature_keys:
            if data[feature_key].shape != feature_shape:
                raise ValueError(f"Feature {feature_key} has shape {data[feature_key].shape} but expected {feature_shape}")
            result += data[feature_key]
        data[self.output_key] = result
        return data


class FeatureInteract(nn.Module):
    """
    A class for interacting between two multidimensional features by reshaping, performing interaction, and reshaping back.
    """
    def __init__(self,
                 feature1_key: str,
                 feature2_key: str,
                 output_key: str):
        super().__init__()
        self.feature1_key = feature1_key
        self.feature2_key = feature2_key
        self.output_key = output_key
        self.model_outputs = [output_key]

        # Weights will be initialized during the forward pass
        self.weights = None

    def forward(self, data: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        feature1 = data[self.feature1_key]  # Shape: [n, A, B, C]
        feature2 = data[self.feature2_key]  # Shape: [n, D, E]

        # Ensure the first dimensions match
        if feature1.shape[0] != feature2.shape[0]:
            raise ValueError(f"Feature1 has shape {feature1.shape} but feature2 has shape {feature2.shape}. Shapes must match.")

        # Save the original shape for reshaping back
        original_shape = feature1.shape  # [n, A, B, C]

        # Reshape both features to [n, -1] (flatten all except the first dimension)
        n = feature1.shape[0]  # The first dimension size (n)
        flattened_size1 = feature1.shape[1:].numel()  # Product of A, B, C
        flattened_size2 = feature2.shape[1:].numel()  # Product of D, E
        feature1_reshaped = feature1.view(n, -1)  # [n, A * B * C]
        feature2_reshaped = feature2.view(n, -1)  # [n, D * E]

        # Dynamically initialize weights based on the reshaped feature sizes during the forward pass
        if self.weights is None:
            self.weights = nn.Parameter(torch.randn(flattened_size2, flattened_size1, dtype=torch.get_default_dtype(), device=feature1.device))

        # Perform the interaction using einsum on the reshaped tensors
        interaction_result = feature1_reshaped * torch.einsum('ij,jk->ik', feature2_reshaped, self.weights)

        # Reshape the result back to the original shape [n, A, B, C]
        interaction_result = interaction_result.view(*original_shape)

        # Save the result in the data dictionary
        data[self.output_key] = interaction_result
        return data

