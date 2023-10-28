import numpy as np
from dscribe.descriptors import MBTR
from ase.io.trajectory import Trajectory
from ase.build import bulk
import matplotlib.pyplot as plt

FILENAME_PREFIX = "test_1533618-amminoborane"

decay = 0.5
mbtr = MBTR(
    species=["H", "B", "N"],
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": 0, "max": 0.5, "sigma": 0.01, "n": 50},
        "weighting": {"function": "exp", "scale": decay, "threshold": 1e-3},
    },
    periodic=True,
    flatten=False,
    sparse=False
)

traj = Trajectory(f"./data_archive/{FILENAME_PREFIX}.traj")
mbtr_output = []
potential = []
for atoms in traj[::5]:
    potential.append(atoms.get_potential_energy())

fig = plt.figure(0)
ax = plt.axes()
ax.set_xlabel("Time / fs")
ax.set_ylabel("Potential / eV")
ax.plot(list(range(100)) + [100],
        potential)
plt.show()

for atoms in traj[::5]:
    mbtr_output.append((mbtr.create(atoms)["k2"][0][0][2], mbtr.create(atoms)["k2"][0][2][3]))

fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel("MBTR descriptor (dimension 0, 0, 2) / Å")
ax.set_ylabel("MBTR descriptor (dimension 0, 2, 3) / Å")
ax.set_zlabel("Potential / eV")
ax.scatter([item[0] for item in mbtr_output],
           [item[1] for item in mbtr_output],
           potential,
           color="red")
plt.show()