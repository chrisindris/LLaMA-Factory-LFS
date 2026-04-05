# Copilot Instructions

# General Repository Environment Instructions

We are using AllianceCan (Compute Canada).
You currently have terminal access to the login node, whereas any SLURM script will be run on a compute node.
The login node does not have GPU access, and does not have a SLURM_TMPDIR (temporary directory created only for the SLURM run).

# Using a python environment

If you want to use a python environment on the login node that is identical to the pyhon environment that the compute node version of the code will see, you can activate it as follows:
`module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 && module load python/3.12 cuda/12.6 opencv/4.12.0 && module load arrow && source /scratch/indrisch/venv_llamafactory_cu126/bin/activate`