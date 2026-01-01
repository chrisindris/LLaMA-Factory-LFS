#!/bin/bash

# this script will output, for each SQA3D scene, the largest number X among all the X.jpg files in the scene's color/ directory

find /scratch/indrisch/vllm_experiments/data/ScanNet/scans/ -maxdepth 1 -type d -name "scene*" -exec sh -c 'f={}; c=$(ls $f/color/*.jpg | sed '"'"'s/\.jpg$//'"'"' | awk -F'"'"'/'"'"' '"'"'{print $NF}'"'"' | sort -n | tail -n 1); echo $f $c' sh {} \;