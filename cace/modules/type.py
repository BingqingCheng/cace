import torch
import torch.nn as nn
from typing import Sequence, Dict

__all__ = [
    'NodeEncoder',
    'NodeEmbedding',
    'EdgeEncoder',
    'EdgeEncoder_InterIntra',
    'NodeEncoder_with_interpolation',
    'ElementEncoder'
]

from .utils import get_edge_node_type

class ElementEncoder(nn.Module):
    def __init__(self, element_dict=None):
        """
        e.g. 
element_dict = {
    1: [1, 0],
    6: [2, 4],
    8: [2, 6],
    # Add more elements as needed
}
        """
        super().__init__()
        # check if all items in the element_dict have the same length
        if element_dict is None:
            element_dict = electron_distribution_dict

        if len(set(len(v) for v in element_dict.values())) != 1:
            raise ValueError("All items in the element_dict should have the same length.")

        self.max_electrons = self.calculate_max_electrons(element_dict)        
        self.element_dict = self.scale_element_dict(element_dict)
        # the length of the vectors in the element_dict
        self.embedding_dim = len(next(iter(element_dict.values())))

    def calculate_max_electrons(self, element_dict):
        max_electrons = [max(max(column), 1) for column in zip(*element_dict.values())]
        return max_electrons

    def scale_element_dict(self, element_dict):
        scaled_dict = {}
        for element, electrons in element_dict.items():
            scaled_electrons = [electron / max_electron for electron, max_electron in zip(electrons, self.max_electrons)]
            scaled_dict[element] = scaled_electrons
        return scaled_dict

    def forward(self, atomic_numbers) -> torch.Tensor:
        encoded_elements = [self.element_dict[num_tensor.item()] for num_tensor in atomic_numbers]
        return torch.tensor(encoded_elements, dtype=torch.get_default_dtype())

class NodeEncoder(nn.Module):
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.register_buffer("index_map", torch.tensor([zs.index(z) if z in zs else -1 for z in range(max(zs) + 1)], dtype=torch.int64))

    def forward(self, atomic_numbers) -> torch.Tensor:
        device = atomic_numbers.device

        # Directly convert atomic numbers to indices using the precomputed map
        indices = self.index_map[atomic_numbers]

        # raise an error if there are out-of-range atomic numbers
        if (indices < 0).any():
            raise ValueError(f"Atomic numbers out of range: {atomic_numbers[indices < 0]}")

        # Generate one-hot encoding
        one_hot_encoding = self.to_one_hot(indices.unsqueeze(-1), num_classes=self.num_classes, device=device)

        return one_hot_encoding

    def to_one_hot(self, indices: torch.Tensor, num_classes: int, device=torch.device) -> torch.Tensor:
        shape = indices.shape[:-1] + (num_classes,)
        oh = torch.zeros(shape, device=device)

        # scatter_ is the in-place version of scatter
        oh.scatter_(dim=-1, index=indices, value=1)
        return oh

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_classes={self.num_classes})"
        )

class NodeEncoder_with_interpolation(nn.Module):
    """
    cumstom NodeEncoder.
    if the atomic number is within zs, using one-hot encoding, otherwise use interpolation between two nearest zs.
    """
    def __init__(self, zs: Sequence[int]):
        super().__init__()
        self.num_classes = len(zs)
        self.zs = zs

    def forward(self, atomic_numbers: torch.Tensor) -> torch.Tensor:
        device = atomic_numbers.device
        # map atomic numbers to indices
        encoded = torch.zeros(atomic_numbers.shape[0], self.num_classes, device=device, dtype=torch.float32)
        # interpolate between two nearest zs
        for i, z in enumerate(atomic_numbers):
            if z in self.zs:
                encoded[i, self.zs.index(z)] = 1
            else:
                for j in range(len(self.zs)):
                    if z < self.zs[j]:
                        encoded[i, j-1] = (self.zs[j] - z) / (self.zs[j] - self.zs[j-1])
                        encoded[i, j] = (z - self.zs[j-1]) / (self.zs[j] - self.zs[j-1])
                        #print(z, i, j , encoded[i, j-1], encoded[i, j])
                        break
        return encoded

class NodeEmbedding(nn.Module):
    def __init__(self, node_dim:int, embedding_dim:int, trainable=True, random_seed=42):
        super().__init__()
        embedding_weights = torch.Tensor(node_dim, embedding_dim)
        if random_seed is not None:
            torch.manual_seed(random_seed)
        self.reset_parameters(embedding_weights)

        if trainable:
            self.embedding_weights = nn.Parameter(embedding_weights)
        else:
            self.register_buffer("embedding_weights", embedding_weights, dtype=torch.get_default_dtype())

    def reset_parameters(self, embedding_weights):
        nn.init.xavier_uniform_(embedding_weights)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return torch.mm(data, self.embedding_weights)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_classes={self.embedding_weights.shape[0]}, embedding_dim={self.embedding_weights.shape[1]})"
        )

class EdgeEncoder(nn.Module):
    def __init__(self, directed=True):
        super().__init__()
        self.directed = directed

    def forward(self,     
               edge_index: torch.Tensor,  # [2, n_edges]
               node_type: torch.Tensor,  # [n_nodes, n_dims]
               node_type_2: torch.Tensor=None,  # [n_nodes, n_dims]
               *args, **kwargs
               ) -> torch.Tensor:
        # Split the edge tensor into two parts for node1 and node2
        node1, node2 = get_edge_node_type(edge_index, node_type, node_type_2)

        if self.directed:
            # Use batched torch.outer for directed edges
            #encoded_edges = torch.bmm(node1.unsqueeze(2), node2.unsqueeze(1)).flatten(start_dim=1)
            encoded_edges = torch.einsum('ki,kj->kij', node1, node2).flatten(start_dim=1)
        else:
            # Sort node1 and node2 along each edge for undirected edges
            min_node, max_node = torch.min(node1, node2), torch.max(node1, node2)
            #encoded_edges = torch.bmm(min_node.unsqueeze(2), max_node.unsqueeze(1)).flatten(start_dim=1)
            encoded_edges = torch.einsum('ki,kj->kij', min_node, max_node).flatten(start_dim=1)

        return encoded_edges

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(directed={self.directed})"
        )

class EdgeEncoder_InterIntra(nn.Module):
    def __init__(self, intramolecular=True, molecular_index_key="molecular_index"):
        super().__init__()
        self.intramolecular = intramolecular
        self.molecular_index_key = molecular_index_key

    def forward(self,
               edge_index: torch.Tensor,  # [2, n_edges]
               node_type: torch.Tensor,  # [n_nodes, n_dims]
               data: Dict[str, torch.Tensor],
               node_type_2: torch.Tensor=None,  # [n_nodes, n_dims]
               ) -> torch.Tensor: # [n_edges, n_dims**2]
        molecular_index = data[self.molecular_index_key]
        # Split the edge tensor into two parts for node1 and node2
        node1, node2 = get_edge_node_type(edge_index, node_type, node_type_2)
        encoded_edges = torch.einsum('ki,kj->kij', node1, node2).flatten(start_dim=1)
        intra_molecular = torch.eq(molecular_index[edge_index[0]], molecular_index[edge_index[1]])
        if self.intramolecular:
            return encoded_edges * intra_molecular.int()[:, None]
        else:
            return encoded_edges * (~intra_molecular).int()[:, None]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(intramolecular={self.intramolecular})"
        )        

# https://en.wikipedia.org/wiki/Electron_shell
electron_distribution_dict = {
    1: [0, 0, 0, 0, 0, 0, 1],    # Hydrogen
    2: [0, 0, 0, 0, 0, 0, 2],    # Helium
    3: [0, 0, 0, 0, 0, 2, 1],    # Lithium
    4: [0, 0, 0, 0, 0, 2, 2],    # Beryllium
    5: [0, 0, 0, 0, 0, 2, 3],    # Boron
    6: [0, 0, 0, 0, 0, 2, 4],    # Carbon
    7: [0, 0, 0, 0, 0, 2, 5],    # Nitrogen
    8: [0, 0, 0, 0, 0, 2, 6],    # Oxygen
    9: [0, 0, 0, 0, 0, 2, 7],    # Fluorine
    10: [0, 0, 0, 0, 0, 2, 8],   # Neon
    11: [0, 0, 0, 0, 2, 8, 1],   # Sodium
    12: [0, 0, 0, 0, 2, 8, 2],   # Magnesium
    13: [0, 0, 0, 0, 2, 8, 3],   # Aluminum
    14: [0, 0, 0, 0, 2, 8, 4],   # Silicon
    15: [0, 0, 0, 0, 2, 8, 5],   # Phosphorus
    16: [0, 0, 0, 0, 2, 8, 6],   # Sulfur
    17: [0, 0, 0, 0, 2, 8, 7],   # Chlorine
    18: [0, 0, 0, 0, 2, 8, 8],   # Argon
    19: [0, 0, 0, 2, 8, 8, 1],   # Potassium
    20: [0, 0, 0, 2, 8, 8, 2],   # Calcium
    21: [0, 0, 0, 2, 8, 9, 2],   # Scandium
    22: [0, 0, 0, 2, 8, 10, 2],   # Titanium
    23: [0, 0, 0, 2, 8, 11, 2],  # Vanadium
    24: [0, 0, 0, 2, 8, 13, 1],  # Chromium
    25: [0, 0, 0, 2, 8, 13, 2],  # Manganese
    26: [0, 0, 0, 2, 8, 14, 2],  # Iron
    27: [0, 0, 0, 2, 8, 15, 2],  # Cobalt
    28: [0, 0, 0, 2, 8, 16, 2],  # Nickel
    29: [0, 0, 0, 2, 8, 18, 1],  # Copper
    30: [0, 0, 0, 2, 8, 18, 2],  # Zinc
    31: [0, 0, 0, 2, 8, 18, 3],  # Gallium
    32: [0, 0, 0, 2, 8, 18, 4],  # Germanium
    33: [0, 0, 0, 2, 8, 18, 5],  # Arsenic
    34: [0, 0, 0, 2, 8, 18, 6],  # Selenium
    35: [0, 0, 0, 2, 8, 18, 7],  # Bromine
    36: [0, 0, 0, 2, 8, 18, 8],  # Krypton
    37: [0, 0, 2, 8, 18, 8, 1],  # Rubidium
    38: [0, 0, 2, 8, 18, 8, 2],  # Strontium
    39: [0, 0, 2, 8, 18, 9, 2],  # Yttrium
    40: [0, 0, 2, 8, 18, 10, 2],  # Zirconium
    41: [0, 0, 2, 8, 18, 12, 1],  # Niobium
    42: [0, 0, 2, 8, 18, 13, 1],  # Molybdenum
    43: [0, 0, 2, 8, 18, 13, 2],  # Technetium
    44: [0, 0, 2, 8, 18, 15, 1],  # Ruthenium
    45: [0, 0, 2, 8, 18, 16, 1],  # Rhodium
    46: [0, 0, 2, 8, 18, 18, 0],  # Palladium
    47: [0, 0, 2, 8, 18, 18, 1],  # Silver
    48: [0, 0, 2, 8, 18, 18, 2],  # Cadmium
    49: [0, 0, 2, 8, 18, 18, 3],  # Indium
    50: [0, 0, 2, 8, 18, 18, 4],  # Tin
    51: [0, 0, 2, 8, 18, 18, 5],  # Antimony
    52: [0, 0, 2, 8, 18, 18, 6],  # Tellurium
    53: [0, 0, 2, 8, 18, 18, 7],  # Iodine
    54: [0, 0, 2, 8, 18, 18, 8],  # Xenon
    55: [0, 2, 8, 18, 18, 8, 1],  # Cesium
    56: [0, 2, 8, 18, 18, 8, 2],  # Barium
    57: [0, 2, 8, 18, 18, 9, 2],  # Lanthanum
    58: [0, 2, 8, 18, 19, 9, 2],  # Cerium
    59: [0, 2, 8, 18, 21, 8, 2],  # Praseodymium
    60: [0, 2, 8, 18, 22, 8, 2],  # Neodymium
    61: [0, 2, 8, 18, 23, 8, 2],  # Promethium
    62: [0, 2, 8, 18, 24, 8, 2],  # Samarium
    63: [0, 2, 8, 18, 25, 8, 2],  # Europium
    64: [0, 2, 8, 18, 25, 9, 2],  # Gadolinium
    65: [0, 2, 8, 18, 27, 8, 2],  # Terbium
    66: [0, 2, 8, 18, 28, 8, 2],  # Dysprosium
    67: [0, 2, 8, 18, 29, 8, 2],  # Holmium
    68: [0, 2, 8, 18, 30, 8, 2],  # Erbium
    69: [0, 2, 8, 18, 31, 8, 2],  # Thulium
    70: [0, 2, 8, 18, 32, 8, 2],  # Ytterbium
    71: [0, 2, 8, 18, 32, 9, 2],  # Lutetium
    72: [0, 2, 8, 18, 32, 10, 2],  # Hafnium
    73: [0, 2, 8, 18, 32, 11, 2],  # Tantalum
    74: [0, 2, 8, 18, 32, 12, 2],  # Tungsten
    75: [0, 2, 8, 18, 32, 13, 2],  # Rhenium
    76: [0, 2, 8, 18, 32, 14, 2],  # Osmium
    77: [0, 2, 8, 18, 32, 15, 2],  # Iridium
    78: [0, 2, 8, 18, 32, 17, 1],  # Platinum
    79: [0, 2, 8, 18, 32, 18, 1],  # Gold
    80: [0, 2, 8, 18, 32, 18, 2],  # Mercury
    81: [0, 2, 8, 18, 32, 18, 3],  # Thallium
    82: [0, 2, 8, 18, 32, 18, 4],  # Lead
    83: [0, 2, 8, 18, 32, 18, 5],  # Bismuth
    84: [0, 2, 8, 18, 32, 18, 6],  # Polonium
    85: [0, 2, 8, 18, 32, 18, 7],  # Astatine
    86: [0, 2, 8, 18, 32, 18, 8],  # Radon
    87: [2, 8, 18, 32, 18, 8, 1],  # Francium
    88: [2, 8, 18, 32, 18, 8, 2],  # Radium
    89: [2, 8, 18, 32, 18, 9, 2],  # Actinium
    90: [2, 8, 18, 32, 18, 10, 2],  # Thorium
    91: [2, 8, 18, 32, 20, 9, 2],  # Protactinium
    92: [2, 8, 18, 32, 21, 9, 2],  # Uranium
    93: [2, 8, 18, 32, 22, 9, 2],  # Neptunium
    94: [2, 8, 18, 32, 24, 8, 2],  # Plutonium
    # Add more elements as needed
}

