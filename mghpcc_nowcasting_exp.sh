#!/bin/sh

# JOB Name 1GPUTest
#BSUB -J 2GPUtest

# GPU queue, using one GPU
#BSUB -q gpu -a gpuexcl_p

# 1GB RAM
#BSUB -R rusage[mem=4096]

# Wall time of 60 minutes
#BSUB -W 10:00

#BSUB -o "/home/an67a/experiment_1.out"
#BSUB -e "/home/an67a/experiment_1.err"

# Load CUDA and Theano modules
module load nvidia_driver/331.38
module load cuda/7.0.28
module load cudnn/v3_for_cuda_7.0
module load python/2.7.9_packages/theano/0.7.0

# our executable 
python gpu_test.py
python -c "from theano.sandbox.cuda.dnn import dnn_available as d; print(d() or d.msg)"
python -c "from lasagne.layers import dnn"

