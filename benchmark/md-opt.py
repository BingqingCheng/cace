import sys
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import time

# --- CONFIGURATION ---
import torch._functorch.config
torch._functorch.config.donated_buffer = False

import torch._inductor.config

# Prevent creation of new graphs for each neighbor list change
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True 

import cace
from cace.representations.cace_representation import Cace
from cace.calculators import CACECalculator

from ase import units
from ase.md.npt import NPT
from ase.io import read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

# --- SETUP ---
torch.serialization.safe_globals([cace.models.atomistic.NeuralNetworkPotential])
torch.set_default_dtype(torch.float32)
torch.set_float32_matmul_precision('medium')

small_conf = read('liquid-64.xyz', '0')
r_factor = 10
init_conf = small_conf.repeat((2, 2, r_factor))
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load Model
cace_nnp = torch.load('best_model.pth', weights_only=False, map_location=device)

# --- CRITICAL FIXES ---

def optimize_model_hybrid(model, atoms):
    print("Applying Hybrid Optimization...")
    box_lengths = atoms.cell.lengths()
    
    ewald_found = False
    
    for module in model.modules():
        name = module.__class__.__name__
        
        # 1. FIX EWALD: Disable compilation for this specific module
        # The Ewald summation uses complex numbers (Fourier space), which crashes 
        # torch.compile. We force it to run in standard PyTorch mode.
        if 'Ewald' in name:
            print(f"  -> Found {name}: Disabling compilation (Complex numbers detected)")
            module.forward = torch._dynamo.disable(module.forward)
            
            # Optional: Lock grid size to improve standard performance
            if hasattr(module, 'dl'):
                dl = module.dl
                static_Nk = [max(1, int(l / dl) + 1) for l in box_lengths]
                module.static_Nk = static_Nk
                print(f"  -> Locked {name} grid size to {static_Nk}")
            
            ewald_found = True

    if not ewald_found:
        print("  -> No Ewald module found (Pure NN model?)")
    
    print("  -> The rest of the model (Neural Network) will be compiled.")

# Apply the hybrid fix
optimize_model_hybrid(cace_nnp, init_conf)

# Compile the whole container. 
# Because we tagged the Ewald module with @disable, 
# torch.compile will optimize the NN and skip the Ewald automatically.
print("Compiling Main Model...")
cace_nnp = torch.compile(cace_nnp, mode='reduce-overhead')

# ----------------------

calculator = CACECalculator(
    model_path=cace_nnp, 
    device=device, 
    energy_key='CACE_energy', 
    forces_key='CACE_forces',
    compute_stress=False,
)

init_conf.set_calculator(calculator)

temperature = 300.0 
MaxwellBoltzmannDistribution(init_conf, temperature * units.kB)

dyn = NPT(
    init_conf, 
    timestep=1 * units.fs, 
    temperature_K=temperature,
    ttime=100 * units.fs, 
    pfactor=None, 
    externalstress=0.0
)

print("Running Warmup...")
dyn.run(10) 
print("Warmup complete.")

start_time = time.time()
dyn.run(100)
end_time = time.time()

sys_size = len(init_conf)
elapsed_time = end_time - start_time
time_per_step = (elapsed_time / 100) * 1000 

print("-" * 50)
print(f"!!! System Size: {sys_size} atoms")
print(f"!!! Total Time:  {elapsed_time:.4f} seconds")
print(f"!!! Performance: {time_per_step:.2f} ms/step")
print("-" * 50)
