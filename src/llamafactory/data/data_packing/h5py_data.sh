#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
<<<<<<< HEAD

# RORQUAL:
#SBATCH --cpus-per-task=64
#SBATCH --time=0-00:20:00
#SBATCH --mem=256GB
#SBATCH --output=%N-h5py_data-%j.out

# TRILLIUM:
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:15:00
#SBATCH --output=%N-h5py_data-%j.out

if [[ "$PS1" == *"rorqual"* ]] || [[ "$HOSTNAME" == *"rorqual"* ]] || [[ "$PS1" == *"rg"* ]] || [[ "$HOSTNAME" == *"rg"* ]]; then
    CLUSTER="RORQUAL"
elif [[ "$PS1" == *"trig"* ]] || [[ "$HOSTNAME" == *"trig"* ]]; then
    CLUSTER="TRILLIUM"
elif [[ "$PS1" == *"klogin"* ]] || [[ "$HOSTNAME" == *"klogin"* ]] || [[ "$PS1" == *"kn"* ]] || [[ "$HOSTNAME" == *"kn"* ]]; then
    CLUSTER="KILLARNEY"
else
    echo "Warning: Could not detect cluster from PS1 or HOSTNAME. Defaulting to RORQUAL."
    CLUSTER="RORQUAL"
fi

module load StdEnv/2023  gcc/12.3  openmpi/4.1.5 
module load python/3.12 cuda/12.6 opencv/4.12.0 
module load arrow
source /scratch/indrisch/venv_llamafactory_cu126/bin/activate

export OMP_NUM_THREADS=1

if [[ "$CLUSTER" == "RORQUAL" ]]; then

    mkdir -p /scratch/indrisch/data_h5py/ScanNet/scans/
    chmod -R 777 /scratch/indrisch/data_h5py/ScanNet/scans/

    python h5py_data.py --input_dir /scratch/indrisch/data/ScanNet/scans/ --output_dir /scratch/indrisch/data_h5py/ScanNet/scans/ --verbose --num_workers 32

elif [[ "$CLUSTER" == "TRILLIUM" ]]; then

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
else
    echo "Invalid cluster: $CLUSTER"
    exit 1
fi
=======
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
>>>>>>> 7a3149e4 (llamafactory can use h5py-packed SQA3D or regular SQA3D)
