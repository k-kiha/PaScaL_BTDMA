FC        = mpifort
NVCC      ?= nvcc
MPICXX    ?= mpicxx
CUDA_ARCH ?= 90
OPT       ?= -O3

FORTRAN_DIR := ../Fortran_Original
CXX_DIR     := ../CUDA_CXX_Port

FORTRAN_GPU_ARCH ?= cc$(CUDA_ARCH)
FORTRAN_CUDAFLAG ?= -cuda -gpu=$(FORTRAN_GPU_ARCH)
FFLAGS_FORTRAN ?= -Mfree -cpp $(OPT) $(FORTRAN_CUDAFLAG) -I$(FORTRAN_DIR)/include/gpu
LDFLAGS_FORTRAN ?= $(FORTRAN_CUDAFLAG)

GENCODE := -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH)
NVCCFLAGS := -std=c++17 $(OPT) $(GENCODE) -ccbin $(MPICXX) -I$(CXX_DIR)/include

FORTRAN_LIB := $(FORTRAN_DIR)/lib/libPaScaL_BTDMA_cuda.a
CXX_LIB     := $(CXX_DIR)/lib/libpascal_btdma_cuda.a

FORTRAN_SRC := example_fortran_btdma_profile.f90
CXX_SRC     := example_cuda_cxx_btdma_profile.cu
FORTRAN_OBJ := .example_fortran_btdma_profile.o

FORTRAN_EXE := example_fortran_btdma_profile
CXX_EXE     := example_cuda_cxx_btdma_profile

.PHONY: all libs clean veryclean help

all: $(FORTRAN_EXE) $(CXX_EXE)

libs: $(FORTRAN_LIB) $(CXX_LIB)

$(FORTRAN_LIB):
	$(MAKE) -C $(FORTRAN_DIR) lib FC="$(FC)" GPU_ARCH="$(FORTRAN_GPU_ARCH)"

$(CXX_LIB):
	$(MAKE) -C $(CXX_DIR) lib CUDA_ARCH=$(CUDA_ARCH) NVCC="$(NVCC)" MPICXX="$(MPICXX)"

$(FORTRAN_EXE): $(FORTRAN_SRC) $(FORTRAN_LIB)
	$(FC) $(FFLAGS_FORTRAN) -c $(FORTRAN_SRC) -o $(FORTRAN_OBJ)
	$(FC) $(LDFLAGS_FORTRAN) $(FORTRAN_OBJ) $(FORTRAN_LIB) -o $@
	rm -f $(FORTRAN_OBJ) *.mod

$(CXX_EXE): $(CXX_SRC) $(CXX_LIB)
	$(NVCC) $(NVCCFLAGS) $< $(CXX_LIB) -o $@

clean:
	rm -f *.o *.mod .*.o

veryclean: clean
	rm -f $(FORTRAN_EXE) $(CXX_EXE)

help:
	@echo "PaScaL_BTDMA matched Study drivers"
	@echo "  make all       build Fortran and CUDA C++ BTDMA profile drivers"
	@echo "  make libs      build original Fortran and CUDA C++ libraries"
	@echo "  make clean     remove transient object/module files"
	@echo "  make veryclean remove transient files and future Study executables"
