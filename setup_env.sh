#!/bin/bash

# Example build/run environment for PaScaL_BTDMA.
# This helper is intended for the example programs in examples/.
# It assumes NVIDIA HPC SDK + CUDA-aware MPI on an NVIDIA GPU system.
# Edit these module names if your cluster uses a different NVIDIA module stack.

echo "Setting PaScaL_BTDMA example build/run environment"

module purge
module load cmake/4.0.2 nvidia-hpc-sdk-hpcx/25.3 cuda/12.8

export MPICH_GPU_SUPPORT_ENABLED=1

module list
