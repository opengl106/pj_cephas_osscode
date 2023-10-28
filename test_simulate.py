from ase.visualize import view
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.andersen import Andersen
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import QuasiNewton
from ase import units
from gpaw import GPAW, PW

"""set input file. """
FILENAME_PREFIX = "1533618-amminoborane"

"""read input files. """
atoms = read(f"./inputs/{FILENAME_PREFIX}.cif")
# atoms = read(f"./inputs/{FILENAME_PREFIX}.xyz")
# view(atoms)

"""set DFT energy function. """
calc = GPAW(mode='lcao', xc='PBE', txt=f'./outputs/test_{FILENAME_PREFIX}.txt')
# calc = GPAW(mode="fd", xc='PBE', txt=f'./outputs/test_{FILENAME_PREFIX}.txt')
# calc = GPAW(mode=PW(340), kpts=(2, 2, 2), xc='PBE', txt=f'./outputs/test_{FILENAME_PREFIX}.txt')
atoms.calc = calc

"""DFT optimize structure. """
# relax = QuasiNewton(atoms, logfile=f'./outputs/qn_{FILENAME_PREFIX}.log')
# relax.run(fmax=0.05)

"""
    assign initial velocity.
    Warning: Using MBD on charged DFT systems may lead to the broken symmetries of wave function!
"""

# MaxwellBoltzmannDistribution(atoms, temperature_K=480)

"""set MD method. """
"""
dyn = VelocityVerlet(atoms, timestep=0.2 * units.fs,
    trajectory=f'./outputs/test_{FILENAME_PREFIX}.traj', logfile=f'./outputs/test_{FILENAME_PREFIX}.log', loginterval=1)

dyn = Andersen(atoms, timestep=0.2 * units.fs,
    temperature_K=500, andersen_prob=0.0005,
    trajectory=f'./outputs/test_{FILENAME_PREFIX}.traj', logfile=f'./outputs/test_{FILENAME_PREFIX}.log', loginterval=1)
"""
dyn = NVTBerendsen(atoms, timestep=0.2 * units.fs,
    temperature_K=500, taut=1 * 1000 * units.fs,
    trajectory=f'./outputs/test_{FILENAME_PREFIX}.traj', logfile=f'./outputs/test_{FILENAME_PREFIX}.log', loginterval=1)

# dyn.attach(calc.write(f'./outputs/test_{FILENAME_PREFIX}.gpw'), interval=1)
def main():
    dyn.run(500)

if __name__ == "__main__":
    main()