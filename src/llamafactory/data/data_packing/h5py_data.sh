#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:10:00
#SBATCH --mem=495GB
#SBATCH --output=%N-h5py_data-%j.out

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 
module load python/3.12 cuda/12.6 opencv/4.12.0 
module load arrow
source /project/aip-wangcs/indrisch/venv_llamafactory_cu126/bin/activate

export OMP_NUM_THREADS=1
mkdir -p /project/aip-wangcs/shared/data_h5py/ScanNet/scans/
chmod -R 777 /project/aip-wangcs/shared/data_h5py/ScanNet/scans/

python h5py_data.py --input_dir /project/aip-wangcs/shared/data/ScanNet/scans/ --output_dir /project/aip-wangcs/shared/data_h5py/ScanNet/scans/ --verbose --num_workers 32

# to get an image and save it to a file:
# python h5py_data.py --get --output_dir /project/aip-wangcs/shared/data_h5py/ScanNet/scans/ --scene_name scene0000_00 --image_name 0.jpg --output_path ./scene0000_00_0.jpg --verbose