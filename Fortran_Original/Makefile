include Makefile.inc

.PHONY: all lib lib_gpu lib_cpu examples cpu \
        ex_btdma_gpu ex_btdma_cpu ex_heat3d ex_euler3d ex_weno2d \
        dirs clean veryclean help

all: lib examples

lib: lib_gpu

examples: ex_btdma_gpu ex_heat3d ex_euler3d ex_weno2d

cpu: lib_cpu ex_btdma_cpu

dirs:
	mkdir -p $(BUILD_CPU) $(BUILD_GPU) $(INC_CPU) $(INC_GPU) $(LIB_DIR) $(RUN_DIR)

GPU_OBJS := \
	$(BUILD_GPU)/mpiutil.o \
	$(BUILD_GPU)/mod_cudatools.o \
	$(BUILD_GPU)/mod_btdma_gpu_v2.o

CPU_OBJS := \
	$(BUILD_CPU)/mpiutil.o \
	$(BUILD_CPU)/mod_btdma_cpu.o

lib_gpu: $(CUDA_LIB)

lib_cpu: $(CPU_LIB)

$(CUDA_LIB): $(GPU_OBJS) | dirs
	$(AR) rcs $@ $(GPU_OBJS)
	$(RANLIB) $@

$(CPU_LIB): $(CPU_OBJS) | dirs
	$(AR) rcs $@ $(CPU_OBJS)
	$(RANLIB) $@

$(BUILD_GPU)/mpiutil.o: $(SRC_DIR)/mpiutil.f90 | dirs
	$(FC) $(FFLAGS_GPU) -c $< -o $@

$(BUILD_GPU)/mod_cudatools.o: $(SRC_DIR)/mod_cudatools.f90 | dirs
	$(FC) $(FFLAGS_GPU) -c $< -o $@

$(BUILD_GPU)/mod_btdma_gpu_v2.o: $(SRC_DIR)/mod_btdma_gpu_v2.f90 $(BUILD_GPU)/mod_cudatools.o | dirs
	$(FC) $(FFLAGS_GPU) -c $< -o $@

$(BUILD_CPU)/mpiutil.o: $(SRC_DIR)/mpiutil.f90 | dirs
	$(FC) $(FFLAGS_CPU) -c $< -o $@

$(BUILD_CPU)/mod_btdma_cpu.o: $(SRC_DIR)/mod_btdma_cpu.f90 $(BUILD_CPU)/mpiutil.o | dirs
	$(FC) $(FFLAGS_CPU) -c $< -o $@

ex_btdma_gpu: $(RUN_DIR)/btdma_gpu.out

ex_btdma_cpu: $(RUN_DIR)/btdma_cpu.out

ex_heat3d: $(RUN_DIR)/heat3d.out

ex_euler3d: $(RUN_DIR)/euler3d.out

ex_weno2d: $(RUN_DIR)/weno2d.out

$(RUN_DIR)/btdma_gpu.out: $(EXAMPLE_DIR)/btdma_gpu/sample_gpu.f90 $(CUDA_LIB) | dirs
	$(FC) $(FFLAGS_GPU) $(BTDMA_GPU_DEFS) -c $< -o $(BUILD_GPU)/sample_gpu.o
	$(FC) $(LDFLAGS_GPU) $(BUILD_GPU)/sample_gpu.o $(CUDA_LIB) -o $@

$(RUN_DIR)/btdma_cpu.out: $(EXAMPLE_DIR)/btdma_cpu/sample_cpu.f90 $(CPU_LIB) | dirs
	$(FC) $(FFLAGS_CPU) $(BTDMA_CPU_DEFS) -c $< -o $(BUILD_CPU)/sample_cpu.o
	$(FC) $(BUILD_CPU)/sample_cpu.o $(CPU_LIB) $(LDFLAGS_CPU) -o $@

$(RUN_DIR)/heat3d.out: $(EXAMPLE_DIR)/heat3d/heat3d_mpi_gpu.f90 $(CUDA_LIB) | dirs
	$(FC) $(FFLAGS_GPU) $(HEAT3D_DEFS) -c $< -o $(BUILD_GPU)/heat3d_mpi_gpu.o
	$(FC) $(LDFLAGS_GPU) $(BUILD_GPU)/heat3d_mpi_gpu.o $(CUDA_LIB) -o $@

$(RUN_DIR)/euler3d.out: $(EXAMPLE_DIR)/euler3d/euler3d_mpi_gpu.f90 $(CUDA_LIB) | dirs
	$(FC) $(FFLAGS_GPU) $(EULER3D_DEFS) -c $< -o $(BUILD_GPU)/euler3d_mpi_gpu.o
	$(FC) $(LDFLAGS_GPU) $(BUILD_GPU)/euler3d_mpi_gpu.o $(CUDA_LIB) -o $@

$(RUN_DIR)/weno2d.out: $(EXAMPLE_DIR)/weno2d/weno_mpi_gpu.f90 $(CUDA_LIB) | dirs
	$(FC) $(FFLAGS_GPU) $(WENO2D_DEFS) -c $< -o $(BUILD_GPU)/weno_mpi_gpu.o
	$(FC) $(LDFLAGS_GPU) $(BUILD_GPU)/weno_mpi_gpu.o $(CUDA_LIB) -o $@

clean:
	rm -rf $(BUILD_DIR)

veryclean: clean
	rm -rf $(INCLUDE_DIR) $(LIB_DIR) $(RUN_DIR)

help:
	@echo "PaScaL_BTDMA NVIDIA build targets"
	@echo "  make              build CUDA library and GPU examples"
	@echo "  make lib          build CUDA library"
	@echo "  make examples     build GPU examples"
	@echo "  make cpu          build CPU library and CPU example"
	@echo "  make clean        remove object files"
	@echo "  make veryclean    remove objects, modules, libraries, and executables"
