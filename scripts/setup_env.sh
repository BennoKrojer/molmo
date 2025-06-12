#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)
export HF_HOME=/mnt/research/scratch/bkroje/molmo_data/huggingface
export MOLMO_DATA_DIR=/mnt/research/scratch/bkroje/molmo_data

echo "Environment variables set:"
echo "MOLMO_DATA_DIR=$MOLMO_DATA_DIR"
echo "HF_HOME=$HF_HOME"
echo "PYTHONPATH=$PYTHONPATH"
