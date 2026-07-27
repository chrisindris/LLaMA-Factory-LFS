#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=0-00:30:00
#SBATCH --output=%N-get_data_slurm_nibi-%j.out
#SBATCH --mail-user=christopher.indris@torontomu.ca
#SBATCH --mail-type=ALL

./get_data.sh NIBI