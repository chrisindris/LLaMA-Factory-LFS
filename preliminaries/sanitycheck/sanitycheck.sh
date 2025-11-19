#!/bin/bash

# inside the apptainer; let's see the situation

echo "Hello, world!"
echo "================================================"
echo "The current directory is $(pwd)"
echo "================================================"
echo "The environment variables are: $(env)"
echo "================================================"
echo "The available GPUs are: $(nvidia-smi)"
echo "================================================"
echo "The available CPUs are: $(nproc)"
echo "================================================"
echo "The available memory is: $(free -h)"
echo "================================================"
echo "The available disk space is: $(df -h)"
echo "================================================"
echo "The available network interfaces are: $(ip link show)"
echo "================================================"
echo "The available network routes are: $(ip route show)"
echo "================================================"

echo "hello world!" > /scratch/indrisch/LLaMA-Factory/preliminaries/sanitycheck/hifromthecontainer.txt

llamafactory-cli help
echo "================================================"
llamafactory-cli train -h
echo "================================================"

pip install unsloth