import sys
sys.path.append('../cace/')
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import time

import cace
from cace.representations.cace_representation import Cace
from cace.calculators import CACECalculator



from ase import units
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md import MDLogger
from ase.io import read, write

torch.serialization.safe_globals([cace.models.atomistic.NeuralNetworkPotential])

torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

small_conf = read('liquid-64.xyz', '0')
r_factor = 10
init_conf = small_conf.repeat((2, 2, r_factor))
cace_nnp = torch.load('best_model.pth', weights_only=False)
#,map_location=torch.device('cpu'))

calculator = CACECalculator(model_path=cace_nnp, 
                            device='cuda', 
                            energy_key='CACE_energy', 
                            forces_key='CACE_forces',
                            compute_stress=False,
                           )

init_conf.set_calculator(calculator)

from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

temperature = 300.0 # in K

# Set initial velocities using Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)


def print_energy(a):
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    print('Energy per atom: Epot = %.4feV  Ekin = %.4feV (T=%3.0fK)  '
          'Etot = %.4feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

def write_frame():
        dyn.atoms.write('md_water-T-'+str(temperature)+'.xyz', append=True)

# Define the NPT ensemble
NPTdamping_timescale = 10 * units.fs  # Time constant for NPT dynamics
NVTdamping_timescale = 100 * units.fs  # Time constant for NVT dynamics (NPT includes both)
dyn = NPT(init_conf, timestep=1 * units.fs, temperature_K=temperature,
          ttime=NVTdamping_timescale, pfactor=None, #0.1*NPTdamping_timescale**2,
          externalstress=0.0)

start_time = time.time()

# Your operation here
# ...

dyn.run(100)

end_time = time.time()

sys_size = 64 *3 * 4* r_factor
elapsed_time = end_time - start_time
print(f"!!! System_size[atom]: {sys_size} Time_taken[seconds]: {elapsed_time}")

exit()
