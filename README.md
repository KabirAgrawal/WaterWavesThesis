# WaterWavesThesis

- Parameters
All sets of parameters to simulate the Kelvin wakes in a 2 dimensional plane

- spacetime_surface_height
Spacetime plots for surface height for homogeneous fluids only (Parameters/unif*.txt)

- kelvin_wake_thesis.py
Python code to conduct simulations for associated parameters. Format for usage:

conda activate dedalus
python3 kelvin_wake_thesis.py parameter_file.txt ./
conda deactivate

arg1 = python file for simulation run
arg2 = parameter file name
arg3 = save directory for simulation results

