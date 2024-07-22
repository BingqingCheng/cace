import cace
import pickle
from ase.io import read
import sys

if len(sys.argv) != 3:
    print('Usage: python compute_avg_e0.py xyzfile stride')
    sys.exit()

stride = int(sys.argv[2])

# read the xyz file and compute the average E0s
xyz = read(sys.argv[1], index=slice(0, None, stride))
avge0 = cace.tools.compute_average_E0s(xyz)

print('Average E0s:', avge0)
# save the avge0 dict to a file
with open('avge0.pkl', 'wb') as f:
    pickle.dump(avge0, f)
