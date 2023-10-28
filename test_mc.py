from math import exp
from time import time

from ase.visualize import view
from ase.calculators.kim.kim import KIM
from ase.io.trajectory import Trajectory, TrajectoryWriter
from dscribe.descriptors import MBTR
from ase.io import read, write
import numpy as np
import tensorflow as tf
from tensorflow import keras

from test_emlm import MBTR_PARAMETERS, FILENAME_PREFIX, TrainData

class MC_PARAMETER:
    """
    Here we choose the E_m and the stepsize in case that the acceptance of each step
    is between 0.4 and 0.6.
    This is by deciding the E_m to be -kTln(0.6) and adjusting the stepsize to make
    the energy fluctuation of each step inside a kTln(0.6 / 0.4) size window.
    """
    nu_0 = 1
    E_m = 0.0132 # in eV, and 1 eV equals to 96.485 kJ/mol (Faraday's constant).
    k = 8.617 * 1e-5
    T = 300
    step = 2 * 1e-4

def potential_predictor_generator(mbtr_parameters, filename_prefix):
    mbtr = MBTR(
        species=["H", "B", "N"],
        k2={
            "geometry": {"function": "inverse_distance"},
            "grid": {"min": mbtr_parameters.MIN, "max": mbtr_parameters.MAX, "sigma": mbtr_parameters.SIGMA, "n": mbtr_parameters.N},
            "weighting": {"function": "exp", "scale": mbtr_parameters.DECAY, "threshold": mbtr_parameters.THRESHOLD},
        },
        periodic=True,
        flatten=False,
        sparse=False
    )
    train_data = TrainData()
    train_data.restore_variables(filename_prefix)
    model = keras.models.load_model(f"./outputs/{filename_prefix}/model")
    def predictor(atoms):
        atoms_mbtr = tf.constant(np.array(mbtr.create(atoms, n_jobs=4)["k2"]).flatten())
        input_tensor = train_data.kernel(atoms_mbtr)
        return model.predict(tf.reshape(input_tensor, [1] + input_tensor.shape))[0][0]
    return predictor

predictor = potential_predictor_generator(MBTR_PARAMETERS, FILENAME_PREFIX)

kim_calc = KIM("Sim_LAMMPS_ReaxFF_WeismillerVanDuinLee_2010_BHNO__SM_327381922729_001")
def kim_calc_func(atoms):
    atoms.calc = kim_calc
    return atoms.get_potential_energy()

"""
for frame in traj[::5]:
    print(frame.get_potential_energy())
    print(predictor(frame))
    print(kim_calc_func(frame))
"""

def probability(energy_now, energy_new):
    if energy_new < energy_now:
        return MC_PARAMETER.nu_0 * exp(-MC_PARAMETER.E_m / (MC_PARAMETER.k * MC_PARAMETER.T))
    else:
        return MC_PARAMETER.nu_0 * exp((-MC_PARAMETER.E_m + energy_now - energy_new)
            / (MC_PARAMETER.k * MC_PARAMETER.T))

def metropolis_step(atoms, calc_func, energy_now=None):
    positions_now = np.array(atoms.get_positions())
    if not energy_now:
        energy_now = calc_func(atoms)
    while True:
        ksi = np.random.rand()
        positions_new = positions_now + (np.random.rand(*positions_now.shape) - 0.5) * MC_PARAMETER.step
        atoms.set_positions(positions_new)
        energy_new = calc_func(atoms)
        prob = probability(energy_now, energy_new)
        if prob > ksi:
            return energy_new

def metropolis(atoms, calc_func, step_num, traj_file_path):
    writer = TrajectoryWriter(traj_file_path)
    energy = calc_func(atoms)
    writer.write(atoms, energy=energy)
    for _ in range(step_num):
        energy = metropolis_step(atoms, calc_func, energy_now=energy)
        writer.write(atoms, energy=energy)

def main():
    traj = Trajectory(f"./data_archive/{FILENAME_PREFIX}.traj")
    atoms = traj[-1]
    time_now_0 = time()
    metropolis(atoms, kim_calc_func, 50000, f"./outputs/{FILENAME_PREFIX}_kim_mc.traj")
    time_new_0 = time()
    time_now_1 = time()
    metropolis(atoms, predictor, 50000, f"./outputs/{FILENAME_PREFIX}_my_mc.traj")
    time_new_1 = time()
    print(f"time spent in simulation 0: {time_new_0 - time_now_0}")
    print(f"time spent in simulation 1: {time_new_1 - time_now_1}")

if __name__ == "__main__":
    main()