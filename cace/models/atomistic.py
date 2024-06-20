from typing import Dict, Optional, List

import torch
import torch.nn as nn

from ..modules import Transform
from ..tools import torch_geometric

__all__ = ["AtomisticModel", "NeuralNetworkPotential", "CombinePotential"]


class AtomisticModel(nn.Module):
    """
    Base class for atomistic neural network models.
    """

    def __init__(
        self,
        postprocessors: Optional[List[Transform]] = None,
        do_postprocessing: bool = False,
    ):
        """
        Args:
            postprocessors: Post-processing transforms that may be
                initialized using the `datamodule`, but are not
                applied during training.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__()
        self.do_postprocessing = do_postprocessing
        self.postprocessors = nn.ModuleList(postprocessors)
        self.required_derivatives: Optional[List[str]] = None
        self.model_outputs: Optional[List[str]] = None

    def collect_derivatives(self) -> List[str]:
        self.required_derivatives = None
        required_derivatives = set()
        for m in self.modules():
            if (
                hasattr(m, "required_derivatives")
                and m.required_derivatives is not None
            ):
                required_derivatives.update(m.required_derivatives)
        required_derivatives: List[str] = list(required_derivatives)
        self.required_derivatives = required_derivatives

    def collect_outputs(self) -> List[str]:
        self.model_outputs = None
        model_outputs = set()
        for m in self.modules():
            if hasattr(m, "model_outputs") and m.model_outputs is not None:
                model_outputs.update(m.model_outputs)
        model_outputs: List[str] = list(model_outputs)
        self.model_outputs = model_outputs

    def initialize_derivatives(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        for p in self.required_derivatives:
            if isinstance(data, torch_geometric.Batch): 
                if p in data.to_dict().keys():
                    data[p].requires_grad_(True)
            else:
                if p in data.keys():
                    data[p].requires_grad_(True)
        return data

    def initialize_transforms(self, datamodule):
        for module in self.modules():
            if isinstance(module, Transform):
                module.datamodule(datamodule)

    def postprocess(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.do_postprocessing:
            # apply postprocessing
            for pp in self.postprocessors:
                data = pp(data)
        return data

    def extract_outputs(
        self, data: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        results = {k: data[k] for k in self.model_outputs}
        return results


class NeuralNetworkPotential(AtomisticModel):
    """
    A generic neural network potential class that sequentially applies a list of input
    modules, a representation module and a list of output modules.

    This can be flexibly configured for various, e.g. property prediction or potential
    energy sufaces with response properties.
    """

    def __init__(
        self,
        representation: nn.Module,
        input_modules: List[nn.Module] = None,
        output_modules: List[nn.Module] = None,
        #input_dtype_str: str = "float32",
        postprocessors: Optional[List[Transform]] = None,
        do_postprocessing: bool = False,
    ):
        """
        Args:
            representation: The module that builds representation from data.
            input_modules: Modules that are applied before representation, e.g. to
                modify input or add additional tensors for response properties.
            output_modules: Modules that predict output properties from the
                representation.
            postprocessors: Post-processing transforms that may be initialized using the
                `datamodule`, but are not applied during training.
            input_dtype_str: The dtype of real data.
            do_postprocessing: If true, post-processing is activated.
        """
        super().__init__(
            #input_dtype_str=input_dtype_str,
            postprocessors=postprocessors,
            do_postprocessing=do_postprocessing,
        )
        self.representation = representation
        self.input_modules = nn.ModuleList(input_modules)
        self.output_modules = nn.ModuleList(output_modules)

        self.collect_derivatives()
        self.collect_outputs()

    def forward(self, 
                data: Dict[str, torch.Tensor], 
                training: bool = False, 
                compute_stress: bool = False, 
                compute_virials: bool = False
                ) -> Dict[str, torch.Tensor]:
        # initialize derivatives for response properties
        data = self.initialize_derivatives(data)

        if 'stress' in self.model_outputs or 'CACE_stress' in self.model_outputs:
            compute_stress = True
        for m in self.input_modules:
            data = m(data, compute_stress=compute_stress, compute_virials=compute_virials)

        data = self.representation(data)

        for m in self.output_modules:
            data = m(data, training=training)

        # apply postprocessing (if enabled)
        data = self.postprocess(data)

        results = self.extract_outputs(data)

        return results

class CombinePotential(nn.Module):
    def __init__(
        self,
        potentials: List[nn.Module],
        potential_keys: List[Dict],
        out_keys: List = ['CACE_energy', 'CACE_forces', 'CACE_stress'],
        operation = None,
    ):
        """
        Combine multiple potentials into a single potential.
        Args:
        potentials: List of potentials to combine.
        potential_keys: List of dictionaries with keys for each potential.
                       e.g. [pot1, pot2] where
        pot1 = {'CACE_energy': 'CACE_energy_intra', 
        'CACE_forces': 'CACE_forces_intra',
        }

        pot2 = {'CACE_energy': 'CACE_energy_inter', 
        'CACE_forces': 'CACE_forces_inter',
        }
        out_keys: List of keys to output. Should be present in all potential_keys.
        operation: Operation to combine the outputs of the potentials.
        """
        super().__init__()
        self.models =  nn.ModuleList([potential for potential in potentials])
        self.potential_keys = potential_keys
        self.required_derivatives = []
        for potential in potentials:
            for d in potential.required_derivatives:
                if d not in self.required_derivatives:
                    self.required_derivatives.append(d)
                    
        self.out_keys = []
        for key in out_keys:
            if all(key in potential_key for potential_key in self.potential_keys):
                self.out_keys.append(key)
                               
        if operation is None:
            # Default operation (sum)
            self.operation = self.default_operation
        else:
            self.operation = operation

    def default_operation(self, my_list):
        return torch.stack(my_list).sum(0)

    
    def forward(self,
                data: Dict[str, torch.Tensor],
                training: bool = False,
                compute_stress: bool = False,
                compute_virials: bool = False
                ) -> Dict[str, torch.Tensor]:
        results = {}
        output = {}
        for i, potential in enumerate(self.models):
            result = potential(data, training, compute_stress, compute_virials)
            results[i] = result
            output.update(result)
            
        for key in self.out_keys:
            values = [ results[i][potential_key[key]] for i, potential_key in enumerate(self.potential_keys)]
            if values:
                output[key] = self.operation(values)
        return output
