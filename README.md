# WaterWavesThesis

- Parameters
All sets of parameters to simulate the Kelvin wakes in a 2 dimensional plane

- spacetime_surface_height
Spacetime plots for surface height for homogeneous fluids only (Parameters/unif*.txt)

- kelvin_wake_thesis.py
Python code to conduct simulations for associated parameters. Format for usage:

```pbs
#!/bin/bash

#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=12:00:00
#PBS -m ae
#PBS -j oe
#PBS -o /srv/scratch/z5163206/exp_unif_d1v1_1

module add openmpi/4.0.3
module add fftw/3.3.9
module add python/3.8.3

source dedalus-p383/bin/activate

python3 kelvin_wake_thesis.py parameter_file.txt ./

deactivate
```

arg1 = python file for simulation run
arg2 = parameter file name
arg3 = save directory for simulation results

