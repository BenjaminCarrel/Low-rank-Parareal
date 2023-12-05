# Low-rank Parareal: a low-rank parallel-in-time integrator

## Abstract

The Parareal algorithm of Lions, Maday, and Turinici is a well-known time parallel algorithm for evolution problems. It is based on a Newton-like iteration, with cheap coarse corrections performed sequentially, and expensive fine solves performed in parallel. In this work, we apply Parareal to evolution problems that admit good low-rank approximations and for which the dynamical low-rank approximation (DLRA), proposed by Koch and Lubich, can be used as time stepper. Many discrete integrators for DLRA have recently been proposed, based on splitting the projected vector field or by applying projected Runge--Kutta methods. The cost and accuracy of these methods are mostly governed by the rank chosen for the approximation. We want to use these properties in a new method, that we call low-rank Parareal, in order to obtain a time-parallel DLRA solver for evolution problems. We propose an analysis of the algorithm on affine linear problems and illustrate our results numerically.

## Authors

- [Benjamin Carrel](https://www.carrel.world) (University of Geneva)
- [Martin J. Gander](https://www.unige.ch/~gander/) (University of Geneva)
- [Bart Vandereycken](https://www.unige.ch/math/vandereycken/) (Univerity of Geneva)

## Reference

The paper is now published here:
[Carrel, B., Gander, M. J., & Vandereycken, B. (2023). Low-rank Parareal: a low-rank parallel-in-time integrator. BIT Numerical Mathematics, 63(1), 13.](https://link.springer.com/article/10.1007/s10543-023-00953-3)

To cite it, you can use the following Bibtex template:
```
@article{carrel2023low,
  title={Low-rank Parareal: a low-rank parallel-in-time integrator},
  author={Carrel, Benjamin and Gander, Martin J and Vandereycken, Bart},
  journal={BIT Numerical Mathematics},
  volume={63},
  number={1},
  pages={13},
  year={2023},
  publisher={Springer}
}
```

You may also want to cite the code specifically, which you can do with the DOI number generated on [Zenodo](https://zenodo.org/records/10262105). The following Bibtex template refers to that Zenodo release:
```
@software{benjamin_carrel_2023_10262105,
  author       = {Benjamin Carrel},
  title        = {{BenjaminCarrel/Low-rank-Parareal: Zenodo Release 
                   1.0}},
  month        = dec,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {Zenodo},
  doi          = {10.5281/zenodo.10262105},
  url          = {https://doi.org/10.5281/zenodo.10262105}
}
```


## Installation

The following instructions worked as of 7th March 2022.

First of all, clone this repositery with

```
git clone https://github.com/BenjaminCarrel/Low-rank-Parareal.git
cd Low-rank-Parareal
```


### General

The general installation setting is:

- Clone this repo on your computer
- Install Python (3.10) with the following packages:
  - numpy (1.22)
  - scipy (1.8)
  - matplotlib (3.5)
  - jupyter (1.0)
  - tqdm (4.63)
  - joblib (1.1)
- Add the folder `src` to your python path to be able to import the code inside

Then, you should be able to run the files of the experiments.

### Conda

With your terminal, go to this folder and create a conda environment with

`conda env create --file environment.yml`

Then, activate the environment with

`conda activate low-rank-parareal`

Finally, add the source folder to the path by running the command

`conda develop src`

You can update your environment with

`conda env update --file environment.yml --prune`


### Apple Silicon

Apple Silicon requires a specific manipulation to install the fastest version of numpy. Make sure that your version of conda is compatible with Apple Silicon. This can be done by installing miniforge with homebrew

`brew install miniforge`

Then, create the conda environment with

```
conda create -n low-rank-parareal
conda activate low-rank-parareal
```

Then, install numpy and scipy with fast BLAS

```
conda install numpy scipy "libblas=*=*accelerate"
```

Finally, install other packages and add `src` to the path

```
conda install jupyter tqdm matplotlib conda-build pip
pip install joblib
conda develop src
```

## Experiments

The experiments made in the paper can be reproduced in each notebook.
More experiments are also made for the interested reader.
For convenience, the output are pre-computed, so you don't need to run the code to see the results.

/!\ Beware that the scripts take a long time to run since they are intended to investigate the behavior of the integrator for multiple choices of the parameters. /!\
All parameters are set by default as described in the paper. Therefore, the figures produced should be the same as in the paper.
If one of the notebook does not run on your computer, please send an e-mail to benjamin.carrel@unige.ch








