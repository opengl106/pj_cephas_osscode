from ase.io.trajectory import Trajectory
from gpaw import GPAW
from dscribe.descriptors import MBTR
import numpy as np
import tensorflow as tf

from test_emlm import MBTR_PARAMETERS, FILENAME_PREFIX, TrainData, kernel_generator

calc = GPAW(mode='lcao', xc='PBE')
traj = Trajectory(f"./data_archive/{FILENAME_PREFIX}_my_mc.traj")
# traj_train = traj[:15]
traj_train = traj[:15000]
energy_traj = []
with open(f'./outputs/test_{FILENAME_PREFIX}_2.log', "w+") as output:
    for i, atoms in enumerate(traj_train):
        print(f"Calculating energy of frame {i}th")
        old_energy = atoms.get_potential_energy()
        atoms.calc = calc
        new_energy = atoms.get_potential_energy()
        energy_traj.append(new_energy)
        output.write(f"{old_energy} {new_energy}\n")

mbtr = MBTR(
    species=["H", "B", "N"],
    k2={
        "geometry": {"function": "inverse_distance"},
        "grid": {"min": MBTR_PARAMETERS.MIN, "max": MBTR_PARAMETERS.MAX, "sigma": MBTR_PARAMETERS.SIGMA, "n": MBTR_PARAMETERS.N},
        "weighting": {"function": "exp", "scale": MBTR_PARAMETERS.DECAY, "threshold": MBTR_PARAMETERS.THRESHOLD},
    },
    periodic=True,
    flatten=False,
    sparse=False
)

train_data = TrainData()

frames_mbtr = []
frames_energy = []
for i, atoms in enumerate(traj_train):
    frames_mbtr.append(tf.constant(np.array(mbtr.create(atoms)["k2"]).flatten()))
    frames_energy.append(energy_traj[i])
    if (i + 1) % 50 == 0:
        print(f"Reading trajectory frame {i}th")
train_data_x_prekernel = tf.data.Dataset.from_tensor_slices(frames_mbtr)
train_data_y = tf.data.Dataset.from_tensor_slices(frames_energy)
kernel = kernel_generator(train_data_x_prekernel)
train_data_x = train_data_x_prekernel.map(kernel)
train_data.train_data_x_prekernel = train_data_x_prekernel
train_data.train_data_x = train_data_x
train_data.train_data_y = train_data_y
train_data.save_variables(f"{FILENAME_PREFIX}_2")
