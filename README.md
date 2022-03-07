# Low-rank Parareal: a low-rank parallel-in-time integrator

## Abstract

The Parareal algorithm of Lions, Maday, and Turinici is a well-known time parallel algorithm for evolution problems. It is based on a Newton-like iteration, with cheap coarse corrections performed sequentially, and expensive fine solves performed in parallel.
  In this work, we apply Parareal to evolution problems that admit good low-rank approximations and for which the dynamical low-rank approximation (DLRA), proposed by Koch and Lubich, can be used as time stepper. Many discrete integrators for DLRA have recently been proposed, based on splitting the projected vector field or by applying projected Runge--Kutta methods. The cost and accuracy of these methods are mostly governed by the rank chosen for the approximation. We want to use these properties in a new method, that we call low-rank Parareal, in order to obtain a time-parallel DLRA solver for evolution problems. We propose an analysis of the algorithm on affine linear problems and illustrate our results numerically.

## Authors

- Benjamin Carrel
- Martin Gander
- Bart Vandereycken

## References

(Put link(s) here)

## Installation

The following instructions worked as of 7th March 2022.

### General

The general installation setting is:

- Clone this repo on your computer
- Install Python (3.10) with the following packages:
  - numpy (1.22)
  - scipy (1.8)
  - matplotlib (3.5)
  - jupyter (1.0)
  - tqdm (4.63)
- Add the folder `src` to your python path to be able to import the code inside

Then, you should be able to run the files of the experiments.

### Conda

With your terminal, go to this folder and create a conda environment with

`conda env create --file environment.yml`

Then, activate the environment with

`conda activate low-rank_parareal`

Finally, add the source folder to the path by running the command

`conda develop src`

You can update your environment with

`conda env update --file environment.yml --prune`


### Apple Silicon

Apple Silicon requires a specific manipulation to install the fastest version of numpy. Make sure that your version of conda is compatible with Apple Silicon. This can be done by installing miniforge3 with homebrew

`brew install miniforge3`

Then, create the conda environment with

```
conda create -n low-rank-parareal
conda activate low-rank-parareal
```

Then, install numpy

```
# install numpy optimized for macOS
conda install cython pybind11
pip3 install --no-binary :all: --no-use-pep517 numpy
# for not reinstalling numpy for other packages
conda config --set pip_interop_enabled true
```

Finally, install other packages and add `src` to the path

```
conda install scipy jupyter tqdm matplotlib conda-build
conda develop src
```

## Experiments

The experiments made in the paper can be reproduced as follows.
All parameters are set by default as described in the paper.
Running the files described below should produce the exact same figures as in the paper.

### Lyapunov

The Lyapunov experiments are in the folder `lyapunov`. 

- The plots of singular values and solution come from `lyapunov/singular_values.py`.
- The plots of several coarse ranks and several fine ranks come from `lyapunov/several_ranks.py`
- The plot of several problem sizes comes from `lyapunov/several_sizes.py`
- The plot of several final times comes from `lyapunov/several_times.py`

A nice animation of the algorithm's behaviour is made in `convergence_animation.py`


### Cookie

The Cookie experiments are in the folder `cookie`.

- The plots of singular values come from `cookie/singular_values.py`
- The plots of several ranks come from `cookie/several_ranks.py`

### Riccati

The Riccati experiments are in the folder `riccati`.

- The plots of singular values come from `riccati/singular_values.py`
- The plots of several ranks come from `riccati/several_ranks.py`





