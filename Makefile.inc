# NVIDIA HPC SDK + MPI build settings for PaScaL_BTDMA.
# Edit FC/GPU_ARCH/LIBS_CPU here if your site uses different defaults.

FC       = mpifort
AR       = ar
RANLIB   = ranlib
GPU_ARCH ?= ccnative

OPT ?= -O3

SRC_DIR      := src
EXAMPLE_DIR  := examples
BUILD_DIR    := build
INCLUDE_DIR  := include
LIB_DIR      := lib
RUN_DIR      := run

BUILD_CPU := $(BUILD_DIR)/cpu
BUILD_GPU := $(BUILD_DIR)/gpu
INC_CPU   := $(INCLUDE_DIR)/cpu
INC_GPU   := $(INCLUDE_DIR)/gpu

CUDA_LIB := $(LIB_DIR)/libPaScaL_BTDMA_cuda.a
CPU_LIB  := $(LIB_DIR)/libPaScaL_BTDMA_cpu.a

FFLAGS_CPU := -Mfree -cpp $(OPT) -module $(INC_CPU) -I$(INC_CPU)
FFLAGS_GPU := -Mfree -cpp -cuda -gpu=$(GPU_ARCH) $(OPT) -DGPU_ENABLED -module $(INC_GPU) -I$(INC_GPU)

LDFLAGS_CPU := -llapack -lblas
LDFLAGS_GPU := -cuda -gpu=$(GPU_ARCH)

BTDMA_CPU_DEFS ?= -DN1=64 -DN2=64 -DN3=4096 -DM=5 -DNRUN=5
BTDMA_GPU_DEFS ?= -DN1=128 -DN2=128 -DN3=2048 -DM=5 -DNRUN=5
HEAT3D_DEFS    ?= -DNX_VAL=128 -DNY_VAL=128 -DNZ_VAL=128 -DNP1=2 -DNP2=2 -DNP3=2 -DNRUN=5
EULER3D_DEFS   ?= -DN1=128 -DN2=128 -DN3=128 -DM=5 -DNP1=2 -DNP2=2 -DNP3=2 -DNRUN=5
WENO2D_DEFS    ?= -DN1=252 -DN2=252 -DM=3 -DNP1=2 -DNP2=2 -DNRUN=5
