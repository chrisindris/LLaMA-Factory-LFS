#!/bin/bash
module load python

virtualenv --no-download $SLURM_TMPDIR/ENV
source $SLURM_TMPDIR/ENV/bin/activate

pip install --no-index --upgrade pip
pip install --no-index ray