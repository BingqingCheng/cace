import sys
sys.path.append('../')
import torch
import cace

from cace.modules import EdgeEncoder

encoder = EdgeEncoder(directed=True)
edges = torch.tensor([
    [[0, 1], [0, 1]], 
    [[0, 1], [1, 0]], 
    [[1, 0], [0, 1]],
    [[1, 0], [1, 0]],
])
encoded_edges = edge_coding(edges)
print("edges:", edges)
print(encoded_edges)

encoder = EdgeEncoder(directed=False)
edges = torch.tensor([[[0, 0.2], [0.7, 0]], [[1, 0], [0, 1]],[[1, 0], [1, 0]], [[0.7, 0], [0.0, 0.2]]])
encoded_edges = encoder(edges)
print("edges:", edges)
print(encoded_edges)


