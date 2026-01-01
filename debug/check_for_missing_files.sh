#!/bin/bash

# this script will check to ensure that all of the needed files are present in the data directory.

# find /scratch/indrisch/data/ScanNet/scans/ -maxdepth 1 -type d -name "scene*" -exec sh -c 'd={}; b=$(basename $d); target=$(cat debug/largest_numbered_jpg_on_trillium.txt | grep $b | awk '"'"'{print $NF}'"'"'); echo $b $target' sh {} \; 

# Check for missing files in the data directory
for dir in /scratch/indrisch/data/ScanNet/scans/*; do
    b=$(basename $dir)
    target=$(cat debug/largest_numbered_jpg_on_trillium.txt | grep $b | awk '{print $NF}')

    echo $dir $b $target

    # loop, from 0 to $target counting by 24 and check if the file exists
    for i in $(seq 0 $target 24); do
        if [ ! -f "$dir/color/$i.jpg" ]; then
            echo "Missing file: $dir/color/$i.jpg"
        fi
    done

done