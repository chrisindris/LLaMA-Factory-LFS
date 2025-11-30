#!/bin/bash 

# since tamia doesn't seem to support array jobs

CLUSTER_NAME=$1

./multi_runner.sh 1 $CLUSTER_NAME
sleep 15m
./multi_runner.sh 2 $CLUSTER_NAME
sleep 15m
./multi_runner.sh 3 $CLUSTER_NAME
sleep 15m
./multi_runner.sh 4 $CLUSTER_NAME
sleep 15m
./multi_runner.sh 5 $CLUSTER_NAME
sleep 15m
