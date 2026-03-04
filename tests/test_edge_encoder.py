import torch

from cace.modules import EdgeEncoder


def test_edge_encoder_directed_shape_and_order():
    encoder = EdgeEncoder(directed=True)
    node_type = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    encoded = encoder(edge_index=edge_index, node_type=node_type)
    assert encoded.shape == (2, 4)
    # Directed encoding should differ for opposite direction in this setup.
    assert not torch.allclose(encoded[0], encoded[1])


def test_edge_encoder_undirected_is_symmetric():
    encoder = EdgeEncoder(directed=False)
    node_type = torch.tensor(
        [
            [1.0, 0.2],
            [0.7, 0.0],
        ]
    )
    edge_index = torch.tensor(
        [
            [0, 1],
            [1, 0],
        ],
        dtype=torch.long,
    )

    encoded = encoder(edge_index=edge_index, node_type=node_type)
    assert encoded.shape == (2, 4)
    # Undirected encoding should be identical for i->j and j->i.
    assert torch.allclose(encoded[0], encoded[1])
