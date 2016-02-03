#!/bin/sh

# JOB Name 1GPUTest
#BSUB -J Deep_Nowcaster

# GPU queue, using one GPU
#BSUB -q gpu -a gpuexcl_p

# 1GB RAM
#BSUB -R rusage[mem=8192]

# Wall time of 48 hours
#BSUB -W 100:00

#BSUB -o "/home/an67a/deep_nowcaster/%J.out"
#BSUB -e "/home/an67a/deep_nowcaster/%J.err"

# Load CUDA and Theano modules
module load nvidia_driver/331.38
module load cuda/7.0.28
module load cudnn/v3_for_cuda_7.0
module load python/2.7.9_packages/theano/0.7.0

# our executable 
pwd
python code/DCNN_nowcaster.py 1000 1200 deep
#python -c "from lasagne.layers import dnn"
