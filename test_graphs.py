
from ase.calculators.kim.kim import KIM
from ase.io.trajectory import Trajectory
from dscribe.descriptors import MBTR
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

POINT_NUM = 500

TRAJ_1 = "test_1533618-amminoborane"
TRAJ_2 = "test_1533618-amminoborane_kim_mc"
TRAJ_3 = "test_1533618-amminoborane_my_mc"

from test_emlm import MBTR_PARAMETERS
from test_mc import predictor, kim_calc_func

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

traj_1 = Trajectory(f"./data_archive/{TRAJ_1}.traj")
traj_2 = Trajectory(f"./data_archive/{TRAJ_2}.traj")
traj_3 = Trajectory(f"./data_archive/{TRAJ_3}.traj")
trajs = [traj_1, traj_2, traj_3]

mbtr_data = []
energies_comp_data_1 = [[], [], []]
energies_comp_data_2 = [[], [], []]
with open("./data_archive/test_test_1533618-amminoborane_2.log", "r") as f:
    singlepoint_data = f.readlines()
    for item in singlepoint_data[::int(15000 / POINT_NUM)]:
        energies_comp_data_2[2].append(float(item.split(" ")[0].lstrip()))
        energies_comp_data_2[0].append(float(item.split(" ")[1].rstrip()))

for i, traj in enumerate(trajs):
    frames_mbtr = []
    for atoms in traj[::int(len(traj) / POINT_NUM)]:
        frames_mbtr.append(tf.constant(np.array(mbtr.create(atoms)["k2"]).flatten()))
    mbtr_data.append(frames_mbtr)
    if i == 0:
        for atoms in traj[::int(len(traj) / POINT_NUM)]:
            energies_comp_data_1[0].append(atoms.get_potential_energy())
            energies_comp_data_1[1].append(kim_calc_func(atoms))
            energies_comp_data_1[2].append(predictor(atoms))
    if i == 2:
        for atoms in traj[:15000:int(15000 / POINT_NUM)]:
            energies_comp_data_2[1].append(kim_calc_func(atoms))

mbtr_all_data = tf.concat(mbtr_data, 0)
mbtr_all_mean = tf.reduce_mean(mbtr_all_data, 0)
mbtr_all_data_centered = mbtr_all_data - mbtr_all_mean
S, _, W = tf.linalg.svd(mbtr_all_data_centered)
pca_res = []
for item in mbtr_data:
    pca_res.append(tf.matmul(item - mbtr_all_mean, W))

fig = plt.figure(0)
ax = plt.axes(projection='3d')
ax.set_xlabel("MBTR descriptor (dimension PCA 0) / Å")
ax.set_ylabel("MBTR descriptor (dimension PCA 1) / Å")
ax.set_zlabel("MBTR descriptor (dimension PCA 2) / Å")
ax.scatter([item[0] for item in pca_res[0]],
           [item[1] for item in pca_res[0]],
           [item[2] for item in pca_res[0]],
           color="blue")
ax.scatter([item[0] for item in pca_res[1]],
           [item[1] for item in pca_res[1]],
           [item[2] for item in pca_res[1]],
           color="cyan")
ax.scatter([item[0] for item in pca_res[2]],
           [item[1] for item in pca_res[2]],
           [item[2] for item in pca_res[2]],
           color="red")
plt.show()

fig = plt.figure(1)
ax = plt.axes(projection='3d')
ax.set_xlabel("MBTR descriptor (dimension PCA 0) / Å")
ax.set_ylabel("MBTR descriptor (dimension PCA 1) / Å")
ax.set_zlabel("MBTR descriptor (dimension PCA 2) / Å")
ax.scatter([item[0] for item in pca_res[1]],
           [item[1] for item in pca_res[1]],
           [item[2] for item in pca_res[1]],
           color="cyan")
ax.scatter([item[0] for item in pca_res[2]],
           [item[1] for item in pca_res[2]],
           [item[2] for item in pca_res[2]],
           color="red")
plt.show()

fig = plt.figure(2)
ax = plt.axes()
ax.set_xlabel("DFT energy")
ax.set_ylabel("Models energy")
ax.scatter(energies_comp_data_1[0],
        energies_comp_data_1[1],
        color="cyan")
ax.scatter(energies_comp_data_1[0],
        energies_comp_data_1[2],
        color="red")
plt.show()

fig = plt.figure(3)
ax = plt.axes()
ax.set_xlabel("DFT energy")
ax.set_ylabel("Models energy")
ax.scatter(energies_comp_data_2[0],
        energies_comp_data_2[1],
        color="cyan")
plt.show()

fig = plt.figure(4)
ax = plt.axes()
ax.set_xlabel("DFT energy")
ax.set_ylabel("Models energy")
ax.scatter(energies_comp_data_2[0],
        energies_comp_data_2[2],
        color="red")
plt.show()
