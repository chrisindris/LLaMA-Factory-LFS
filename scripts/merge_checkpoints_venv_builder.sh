#!/bin/bash

module load StdEnv/2023 gcc/12.3 openmpi/4.1.5
module load python/3.12 cuda/12.6 opencv/4.12.0
module load arrow

virtualenv --no-download ENV && source ENV/bin/activate
pip install --upgrade --no-index pip

pip install packaging
pip install --no-index -r /scratch/indrisch/LLaMA-Factory/requirements.txt