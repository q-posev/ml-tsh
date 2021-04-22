# ml-tsh

The `ml-tsh` set of scripts was initially developed as a patch for the molecular dynamics driver of Atomic Simulation Environment (ASE) in order to perform Trajectory Surface Hopping (TSH) simulations. The idea is that one can propagate a classical trajectory on a given PES with a Landau-Zener probability to hop from one state to another. The electronic structure calculations are performed either using TD-DFTB method or using a pre-trained Deep Learning (SchNet) model.

The script is currently adapted for an electronic relaxation from S3 excited state of phenanthrene.

#### Currently provided:

- `tsh-driver.py` : initializes and laucnes a single trajectory with a set of parameters provided from the command line
- `submit-chdb-py.sh` : SLURM submission script to go embarrassingly parallel using chdb `placement` tool at CALMIP computing center (OLYMPE machine)

#### Requirements (versions used at CALMIP):
- python 3 	(3.6.9)
- numpy		(1.17.3)
- pandas	(1.0.3)
- ase		(3.19.0b1)
- schnetpack	(0.3.1)

_**Note: I recommend using virtual environments (e.g. `miniconda`) to avoid compatibility issues.**_

## Clone the repository

```
git clone https://github.com/q-posev/ml-tsh
cd ml-tsh
```

You're ready to go!

## Single trajectory

A single classical trajectory can be launched using:

```
python tsh-driver.py 4000 0.25 schnet lz 3state
```

#### Required arguments:
- #1 is the number of MD steps to perform (e.g. `4000`)
- #2 is the nuclear time step in fs (e.g. `0.25`)
- #3 is the calculator name for the electronic structure part (`schnet` or `demonnano`)
- #4 is the type of a hopping approach for TSH (`lz` or `zn`; `lz` and `zn` correspond to Belyaev-Lebedev and Zhu-Nakamura schemes, respectively)
- #5 is the model type based on the number of states included in the propagation (`3state` or `2state`)

At the moment, the paths to the pre-trained SchNet models are hardcoded and have to be adapted manually in the code according to the paths on your machine.

_**Note 1: A set of initial conditions has to be in the same directory for this script to launch. Geometries and velocities are stored in `geom-phen.xyz` and `velo` files, respectively.**_

_**Note 2: The final number of MD steps in the output file can be different from the input one. This is due to the fact that we actually have to come to the previous MD step if the hop was accepted.**_


## Ensemble of trajectories

To launch an ensemble of trajectories using `submit-chdb-py.sh`:
- modify the parameters for a single trajectory in the line that contains `--command`
- adapt the SLURM parameters to your needs (e.g. number of tasks/trajectories, number of nodes and tasks per node etc.)
- launch the submission script
```
sbatch submit-chdb-py.sh
```

_**Note: (OPTIONAL) You can use additional argument to skip first N initial conditions/trajectories. For example, `sbatch submit-chdb-py.sh add 106` will start taking initial conditions from a set #107.**_

- check that placement is correct using the following command (for CALMIP users)
```
placement --checkme
```

## Post-processing and visualisation

The script required for post-processing and visualisation is provided upon request. It relies on additional Python packages (matplotlib and scipy) to fit and plot the results. Alternatively, one can easily adapt the existing `plot-tsh-occs.py` script (see https://github.com/q-posev/fit_and_plot).

