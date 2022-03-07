# Low-rank Parareal: a low-rank parallel-in-time integrator

## Abstract

The Parareal algorithm is a well-known time parallel algorithm for evolution problems. It is based on a Newton-like iteration, with cheap coarse corrections performed sequentially, and expensive fine solves performed in parallel.
We are interested here in evolution problems that admit good low-rank approximations, and where the dynamical low-rank approximation (DLRA), proposed by Koch and Lubich, can be used as time stepper. Many discrete integrators for DLRA have recently been proposed, like the projector-splitting methods and the projected Runge-Kutta methods. The cost and accuracy in DLRA are mostly governed by the rank chosen for the approximation. We want to use these properties in a new method that we call low-rank Parareal, in order to obtain time parallel DLRA solvers for evolution problems. We propose an analysis of the algorithm on linear problems, and illustrate our results numerically.

## Authors

- Benjamin Carrel
- Martin Gander
- Bart Vandereycken

## References

(Put link(s) here)

## Installation

Clone this folder in your computer.

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
conda create -n low-rank_parareal
conda activate low-rank_parareal
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






