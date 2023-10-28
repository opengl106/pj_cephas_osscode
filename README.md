# Introduction

This project "pj_cephas", by Lena Shizuki (legally named Ziyue Ji), is an experimental project started from 07/2022 and completed (or say, halted) at 09/2022, after an online workshop of molecular dynamics/monte carlo simulation (MD/MC).

## General Idea

The general idea is to prove that distance-based linear machine learning algorithms can also predict energies and design MD/MC forcefields for reactive systems. Although what the arthor did in this project is some experimental toy-like computations, it should be considered enough to prove the feasibility of this research idea, at that time in 2022.

## Writting sample

This is an unpublished work by Ziyue Ji in 09/15/2022 as the (temporarily) result of this project.

[Using distance-based linear machine learning model to learn Monte Carlo simulation potentials on small reactive systems](/writtensample/Using%20distance-based%20linear%20machine%20learning%20model%20to%20learn.pdf)

# Installing deps and envs

## Configure
sudo add-apt-repository ppa:openkim/latest
sudo apt-get update
sudo apt-get install build-essential cmake cmake-curses-gui python3-dev libxc-dev libopenblas-dev libscalapack-mpi-dev libfftw3-dev python3-tk jmol libkim-api-dev openkim-models clang-format ffmpeg
python3 -m venv ./venv
source venv/bin/activate
python -m pip install -r requirements.txt
gpaw install-data /mnt/4058FAC158FAB52E/Program\ Files\ \(x86\)/gpaw

## compile LAMMPS with KIM on
python3 -m venv ./lammps_venv
source lammps_venv/bin/activate
git clone -b develop https://github.com/lammps/lammps.git mylammps
cd mylammps
mkdir build
cd build
cmake -C ../cmake/presets/basic.cmake -D PKG_KIM=yes -D PKG_REAXFF=yes -D DOWNLOAD_KIM=yes -D KIM_EXTRA_UNITTESTS=yes -D BUILD_SHARED_LIBS=on -D LAMMPS_EXCEPTIONS=on -D PKG_PYTHON=on ../cmake
cmake --build .
cmake --install .
make install-python

## install CUDA

see https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local

## also see
https://wiki.fysik.dtu.dk/gpaw/platforms/Linux/ubuntu.html
https://openkim.org/doc/usage/obtaining-models/#ubuntu_linux
