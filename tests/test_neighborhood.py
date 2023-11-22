import sys
sys.path.append('../')
import numpy as np
from ase.io import read
import cace
#from cace import data

atoms = read('../datasets/water.xyz','0')
config = cace.data.config_from_atoms(atoms, energy_key ='TotEnergy', forces_key='force')

edge_index, shifts, unit_shifts = cace.data.get_neighborhood(
                                  positions=config.positions,
                                  cutoff=5,
                                  cell=config.cell,
                                  pbc=config.pbc
                                  )

print(np.shape(edge_index))

# try if the same number of neighbors as the usual method
from ase.neighborlist import NeighborList
# Generate neighbor list with element-specific cutoffs
cutoffs = [2.5 for atom in atoms]
nl = NeighborList(cutoffs, skin=0.0, self_interaction=False, bothways=True)
nl.update(atoms)

# Store displacement vectors
displacement_vectors = []

for i, atom in enumerate(atoms):
    indices, offsets = nl.get_neighbors(i)
    for j, offset in zip(indices, offsets):
        disp_vector = atoms[j].position - atom.position
        disp_vector += np.dot(offset, atoms.get_cell())
        displacement_vectors.append(disp_vector)
        
print(np.shape(displacement_vectors))

assert np.shape(edge_index)[1] == np.shape(displacement_vectors)[0]


