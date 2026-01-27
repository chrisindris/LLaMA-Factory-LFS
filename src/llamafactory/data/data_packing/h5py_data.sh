#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:15:00
#SBATCH --output=%N-h5py_data-%j.out

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 
module load python/3.12 cuda/12.6 opencv/4.12.0 
module load arrow
source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

export OMP_NUM_THREADS=1

mkdir -p /scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/
chmod -R 777 /scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/

# to pack data
python h5py_data.py --input_dir /scratch/indrisch/vllm_experiments/data/ScanNet/scans/ --output_dir /scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/ --verbose --num_workers 32

# to retrieve data
# python h5py_data.py --get --output_dir /scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/ --scene_name scene0000_00 --image_name 0.jpg --output_path ./scene0000_00_0.jpg --verbose

# in Python:
# from h5py_data import *
# b = retrieve_image("/scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/", "scene0000_00", "0.jpg")
# alternatively: 
# b = retrieve_image("/scratch/indrisch/vllm_experiments/data_h5py/ScanNet/scans/scene0000_00/color/0.jpg")
# Image.open(io.BytesIO(b)).save('0.jpg', format="JPEG") # if you want the above bytes object to be saved