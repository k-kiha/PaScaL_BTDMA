CUDA_ARCH ?= 90
FC        = mpifort
NVCC      ?= nvcc
MPICXX    ?= mpicxx

FORTRAN_GPU_ARCH ?= cc$(CUDA_ARCH)

.PHONY: all libs fortran fortran-lib fortran-examples fortran-cpu \
        cuda-cxx cuda-cxx-lib cuda-cxx-example study \
        clean veryclean help

all: libs study

libs: fortran-lib cuda-cxx-lib

fortran: fortran-lib fortran-examples

fortran-lib:
	$(MAKE) -C Fortran_Original lib FC="$(FC)" GPU_ARCH="$(FORTRAN_GPU_ARCH)"

fortran-examples: fortran-lib
	$(MAKE) -C Fortran_Original examples FC="$(FC)" GPU_ARCH="$(FORTRAN_GPU_ARCH)"

fortran-cpu:
	$(MAKE) -C Fortran_Original cpu FC="$(FC)" GPU_ARCH="$(FORTRAN_GPU_ARCH)"

cuda-cxx: cuda-cxx-lib cuda-cxx-example

cuda-cxx-lib:
	$(MAKE) -C CUDA_CXX_Port lib CUDA_ARCH=$(CUDA_ARCH) NVCC="$(NVCC)" MPICXX="$(MPICXX)"

cuda-cxx-example: cuda-cxx-lib
	$(MAKE) -C CUDA_CXX_Port example CUDA_ARCH=$(CUDA_ARCH) NVCC="$(NVCC)" MPICXX="$(MPICXX)"

study: libs
	$(MAKE) -C Study all CUDA_ARCH=$(CUDA_ARCH) FC="$(FC)" NVCC="$(NVCC)" MPICXX="$(MPICXX)"

clean:
	$(MAKE) -C Fortran_Original clean
	$(MAKE) -C CUDA_CXX_Port clean
	$(MAKE) -C Study clean

veryclean:
	$(MAKE) -C Fortran_Original veryclean
	$(MAKE) -C CUDA_CXX_Port veryclean
	$(MAKE) -C Study veryclean

help:
	@echo "PaScaL_BTDMA study-oriented build"
	@echo "  make all              build Fortran/CUDA C++ libraries and Study skeleton"
	@echo "  make libs             build only implementation libraries"
	@echo "  make fortran          build original CUDA Fortran library and GPU examples"
	@echo "  make fortran-cpu      build original CPU library and CPU example"
	@echo "  make cuda-cxx         build CUDA C++ port library and sample"
	@echo "  make study            build Study dependencies"
	@echo "  make clean            remove intermediate build files"
	@echo "  make veryclean        remove generated libraries and executables"
