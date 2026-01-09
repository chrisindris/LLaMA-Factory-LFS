#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:20:00
#SBATCH --mem=256GB
#SBATCH --output=%N-h5py_data-%j.out

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 
module load python/3.12 cuda/12.6 opencv/4.12.0 
module load arrow
source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

export OMP_NUM_THREADS=1

mkdir -p /scratch/indrisch/data_h5py/ScanNet/scans/
chmod -R 777 /scratch/indrisch/data_h5py/ScanNet/scans/

python h5py_data.py --input_dir /scratch/indrisch/data/ScanNet/scans/ --output_dir /scratch/indrisch/data_h5py/ScanNet/scans/ --verbose --num_workers 32