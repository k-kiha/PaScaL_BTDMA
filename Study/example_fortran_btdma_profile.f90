program example_fortran_btdma_profile
    use mpi
    use cudafor
    use mod_btdma_gpu_v2
    implicit none

    integer, parameter :: n_timing_fields = 8

    integer :: ierr, myrank, nprocs
    integer :: ngpu, gpurank
    integer :: n1, n2, n3, m, iterations
    integer :: z_first, z_last, nrow, nsys
    integer :: nrow_min, nrow_max, iter
    real(8) :: phase_local(n_timing_fields)
    real(8) :: phase_max(n_timing_fields), phase_sum(n_timing_fields)

    real(8), allocatable, dimension(:,:,:), device :: D_d
    real(8), allocatable, dimension(:,:,:,:), device :: A_d, B_d, C_d
    real(8), allocatable, dimension(:,:,:) :: D_h

    type(BTDMA_PLAN_gpu_v2) :: plan
    type(BTDMA_TIMING_gpu_v2) :: timing

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    call parse_positive_arg(1, 32, n1, "n1", myrank)
    call parse_positive_arg(2, 32, n2, "n2", myrank)
    call parse_positive_arg(3, 128, n3, "n3", myrank)
    call parse_positive_arg(4, 5, m, "m", myrank)
    call parse_positive_arg(5, 10, iterations, "iterations", myrank)

    if (command_argument_count() > 5) then
        if (myrank == 0) then
            write(*,*) "usage: example_fortran_btdma_profile [n1] [n2] [n3] [m] [iterations]"
        endif
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif

    if (m > 8) then
        if (myrank == 0) write(*,*) "m must be <= 8 for the current GPU BTDMA kernels."
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif

    ierr = cudaGetDeviceCount(ngpu)
    if (ngpu <= 0) then
        if (myrank == 0) write(*,*) "No CUDA device is visible."
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif
    gpurank = mod(myrank, ngpu)
    ierr = cudaSetDevice(gpurank)
    ierr = cudaDeviceSynchronize()

    call partition_1d(0, n3 - 1, nprocs, myrank, z_first, z_last)
    nrow = z_last - z_first + 1
    nsys = n1 * n2

    call MPI_REDUCE(nrow, nrow_min, 1, MPI_INTEGER, MPI_MIN, 0, MPI_COMM_WORLD, ierr)
    call MPI_REDUCE(nrow, nrow_max, 1, MPI_INTEGER, MPI_MAX, 0, MPI_COMM_WORLD, ierr)
    if (nrow < 2) then
        if (myrank == 0) write(*,*) "Each rank needs at least two local rows."
        call MPI_ABORT(MPI_COMM_WORLD, 1, ierr)
    endif

    allocate(A_d(1:nsys,1:nrow,1:m,1:m))
    allocate(B_d(1:nsys,1:nrow,1:m,1:m))
    allocate(C_d(1:nsys,1:nrow,1:m,1:m))
    allocate(D_d(1:nsys,1:nrow,1:m))
    allocate(D_h(1:nsys,1:nrow,1:m))

    call btdma_makeplan_gpu_v2(plan, m, nsys, nrow, MPI_COMM_WORLD)

    if (myrank == 0) then
        write(*,'(A)',advance='no') "solver,variant,implementation,nranks,n1,n2,n3,m,nsys,"
        write(*,'(A)',advance='no') "nrow_min,nrow_max,iter,iterations,mpi_mode,total_s_max,total_s_avg,"
        write(*,'(A)',advance='no') "local_compute_s_max,forward_exchange_s_max,reduced_compute_s_max,"
        write(*,'(A)',advance='no') "backward_exchange_s_max,update_compute_s_max,compute_s_max,"
        write(*,'(A)') "communication_s_max"
    endif

    do iter = 0, iterations - 1
        call initialize_coefficients(A_d, B_d, C_d, D_d, nsys, nrow, m)
        ierr = cudaDeviceSynchronize()

        call MPI_BARRIER(MPI_COMM_WORLD, ierr)
        call btdma_many_mpi_gpu_v2_profiled(A_d, B_d, C_d, D_d, m, nsys, nrow, plan, timing)

        if (iter == 0) then
            D_h = D_d
            call write_signature_csv_if_requested(D_h, n1, n2, n3, m, nsys, nrow, &
                                                  nrow_min, nrow_max, z_first, z_last, &
                                                  nprocs, myrank)
        endif

        call fill_timing_fields(timing, phase_local)
        call MPI_REDUCE(phase_local, phase_max, n_timing_fields, MPI_DOUBLE_PRECISION, &
                        MPI_MAX, 0, MPI_COMM_WORLD, ierr)
        call MPI_REDUCE(phase_local, phase_sum, n_timing_fields, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, 0, MPI_COMM_WORLD, ierr)

        if (myrank == 0) then
            call write_csv_row(nprocs, n1, n2, n3, m, nsys, nrow_min, nrow_max, &
                               iter, iterations, phase_max, phase_sum(1) / dble(nprocs))
        endif
    enddo

    call btdma_cleanplan_gpu_v2(plan)
    deallocate(A_d, B_d, C_d, D_d, D_h)
    call MPI_FINALIZE(ierr)

contains

    subroutine parse_positive_arg(index, default_value, value, name, rank)
        integer, intent(in) :: index, default_value, rank
        integer, intent(out) :: value
        character(len=*), intent(in) :: name
        character(len=64) :: arg
        integer :: stat, ierr_local

        value = default_value
        if (command_argument_count() >= index) then
            call get_command_argument(index, arg)
            read(arg, *, iostat=stat) value
            if (stat /= 0 .or. value <= 0) then
                if (rank == 0) write(*,*) trim(name), " must be a positive integer."
                call MPI_ABORT(MPI_COMM_WORLD, 1, ierr_local)
            endif
        endif
    end subroutine parse_positive_arg

    subroutine partition_1d(start_index, end_index, nprocs_arg, rank_arg, first, last)
        integer, intent(in) :: start_index, end_index, nprocs_arg, rank_arg
        integer, intent(out) :: first, last
        integer :: n, base, rem

        n = end_index - start_index + 1
        base = n / nprocs_arg
        rem = mod(n, nprocs_arg)
        first = rank_arg * base + start_index + min(rank_arg, rem)
        last = first + base - 1
        if (rem > rank_arg) last = last + 1
    end subroutine partition_1d

    subroutine initialize_coefficients(A, B, C, D, nsys_arg, nrow_arg, m_arg)
        integer, intent(in) :: nsys_arg, nrow_arg, m_arg
        real(8), device, intent(inout) :: A(1:nsys_arg,1:nrow_arg,1:m_arg,1:m_arg)
        real(8), device, intent(inout) :: B(1:nsys_arg,1:nrow_arg,1:m_arg,1:m_arg)
        real(8), device, intent(inout) :: C(1:nsys_arg,1:nrow_arg,1:m_arg,1:m_arg)
        real(8), device, intent(inout) :: D(1:nsys_arg,1:nrow_arg,1:m_arg)
        integer :: p

        A = 0.0d0
        B = 0.0d0
        C = 0.0d0
        D = 1.0d0

        do p = 1, m_arg
            A(:,:,p,p) = 0.25d0
            B(:,:,p,p) = 2.00d0
            C(:,:,p,p) = 0.25d0
        enddo
    end subroutine initialize_coefficients

    subroutine fill_timing_fields(timing_arg, values)
        type(BTDMA_TIMING_gpu_v2), intent(in) :: timing_arg
        real(8), intent(out) :: values(n_timing_fields)

        values(1) = timing_arg%total
        values(2) = timing_arg%local_compute
        values(3) = timing_arg%forward_exchange
        values(4) = timing_arg%reduced_compute
        values(5) = timing_arg%backward_exchange
        values(6) = timing_arg%update_compute
        values(7) = timing_arg%local_compute + timing_arg%reduced_compute + timing_arg%update_compute
        values(8) = timing_arg%forward_exchange + timing_arg%backward_exchange
    end subroutine fill_timing_fields

    subroutine write_csv_row(nprocs_arg, n1_arg, n2_arg, n3_arg, m_arg, nsys_arg, &
                             nrow_min_arg, nrow_max_arg, iter_arg, iterations_arg, &
                             phase_max_arg, total_avg_arg)
        integer, intent(in) :: nprocs_arg, n1_arg, n2_arg, n3_arg, m_arg, nsys_arg
        integer, intent(in) :: nrow_min_arg, nrow_max_arg, iter_arg, iterations_arg
        real(8), intent(in) :: phase_max_arg(n_timing_fields), total_avg_arg
        integer :: i

        write(*,'(A)',advance='no') "btdma,noncyclic,fortran-original,"
        write(*,'(I0,A)',advance='no') nprocs_arg, ","
        write(*,'(I0,A)',advance='no') n1_arg, ","
        write(*,'(I0,A)',advance='no') n2_arg, ","
        write(*,'(I0,A)',advance='no') n3_arg, ","
        write(*,'(I0,A)',advance='no') m_arg, ","
        write(*,'(I0,A)',advance='no') nsys_arg, ","
        write(*,'(I0,A)',advance='no') nrow_min_arg, ","
        write(*,'(I0,A)',advance='no') nrow_max_arg, ","
        write(*,'(I0,A)',advance='no') iter_arg, ","
        write(*,'(I0,A)',advance='no') iterations_arg, ","
        write(*,'(A)',advance='no') "device,"
        write(*,'(ES24.16,A)',advance='no') phase_max_arg(1), ","
        write(*,'(ES24.16,A)',advance='no') total_avg_arg, ","
        do i = 2, n_timing_fields - 1
            write(*,'(ES24.16,A)',advance='no') phase_max_arg(i), ","
        enddo
        write(*,'(ES24.16)') phase_max_arg(n_timing_fields)
    end subroutine write_csv_row

    subroutine write_signature_csv_if_requested(solution, n1_arg, n2_arg, n3_arg, &
                                                m_arg, nsys_arg, nrow_arg, &
                                                nrow_min_arg, nrow_max_arg, &
                                                z_first_arg, z_last_arg, &
                                                nprocs_arg, rank_arg)
        integer, intent(in) :: n1_arg, n2_arg, n3_arg, m_arg, nsys_arg, nrow_arg
        integer, intent(in) :: nrow_min_arg, nrow_max_arg, z_first_arg, z_last_arg
        integer, intent(in) :: nprocs_arg, rank_arg
        real(8), intent(in) :: solution(1:nsys_arg,1:nrow_arg,1:m_arg)
        character(len=512) :: output_path
        integer :: path_len, env_status, ierr_local
        integer :: sample_z(3), idx, unit, io_status, file_size
        logical :: file_exists, need_header
        real(8) :: local_sum, local_sumsq, local_linf
        real(8) :: global_sum, global_sumsq, global_linf
        real(8) :: local_samples(3), global_samples(3)

        call get_environment_variable("PASCAL_BTDMA_SIGNATURE_OUT", output_path, &
                                      length=path_len, status=env_status)

        local_sum = sum(solution)
        local_sumsq = sum(solution * solution)
        local_linf = maxval(abs(solution))

        sample_z = (/ 0, n3_arg / 2, n3_arg - 1 /)
        local_samples = 0.0d0
        do idx = 1, 3
            if (sample_z(idx) >= z_first_arg .and. sample_z(idx) <= z_last_arg) then
                local_samples(idx) = solution(1, sample_z(idx) - z_first_arg + 1, 1)
            endif
        enddo

        call MPI_REDUCE(local_sum, global_sum, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                        0, MPI_COMM_WORLD, ierr_local)
        call MPI_REDUCE(local_sumsq, global_sumsq, 1, MPI_DOUBLE_PRECISION, MPI_SUM, &
                        0, MPI_COMM_WORLD, ierr_local)
        call MPI_REDUCE(local_linf, global_linf, 1, MPI_DOUBLE_PRECISION, MPI_MAX, &
                        0, MPI_COMM_WORLD, ierr_local)
        call MPI_REDUCE(local_samples, global_samples, 3, MPI_DOUBLE_PRECISION, &
                        MPI_SUM, 0, MPI_COMM_WORLD, ierr_local)

        if (rank_arg /= 0) return
        if (env_status /= 0 .or. path_len <= 0) return

        file_size = 0
        inquire(file=trim(output_path(1:path_len)), exist=file_exists, size=file_size)
        need_header = (.not. file_exists) .or. (file_size == 0)
        open(newunit=unit, file=trim(output_path(1:path_len)), status="unknown", &
             position="append", action="write", iostat=io_status)
        if (io_status /= 0) then
            write(*,*) "Failed to open solution signature CSV: ", trim(output_path(1:path_len))
            call MPI_ABORT(MPI_COMM_WORLD, 1, ierr_local)
        endif

        if (need_header) call write_signature_header(unit)
        call write_signature_row(unit, nprocs_arg, n1_arg, n2_arg, n3_arg, m_arg, &
                                 nsys_arg, nrow_min_arg, nrow_max_arg, global_sum, &
                                 global_sumsq, global_linf, global_samples)
        close(unit)
    end subroutine write_signature_csv_if_requested

    subroutine write_signature_header(unit)
        integer, intent(in) :: unit

        write(unit,'(A)',advance='no') "solver,variant,implementation,nranks,n1,n2,n3,m,nsys,"
        write(unit,'(A)',advance='no') "nrow_min,nrow_max,mpi_mode,solution_sum,"
        write(unit,'(A)',advance='no') "solution_l2,solution_linf,sample_z0,"
        write(unit,'(A)') "sample_zmid,sample_zlast"
    end subroutine write_signature_header

    subroutine write_signature_row(unit, nprocs_arg, n1_arg, n2_arg, n3_arg, m_arg, &
                                   nsys_arg, nrow_min_arg, nrow_max_arg, &
                                   global_sum_arg, global_sumsq_arg, &
                                   global_linf_arg, global_samples_arg)
        integer, intent(in) :: unit, nprocs_arg, n1_arg, n2_arg, n3_arg
        integer, intent(in) :: m_arg, nsys_arg, nrow_min_arg, nrow_max_arg
        real(8), intent(in) :: global_sum_arg, global_sumsq_arg, global_linf_arg
        real(8), intent(in) :: global_samples_arg(3)

        write(unit,'(A)',advance='no') "btdma,noncyclic,fortran-original,"
        write(unit,'(I0,A)',advance='no') nprocs_arg, ","
        write(unit,'(I0,A)',advance='no') n1_arg, ","
        write(unit,'(I0,A)',advance='no') n2_arg, ","
        write(unit,'(I0,A)',advance='no') n3_arg, ","
        write(unit,'(I0,A)',advance='no') m_arg, ","
        write(unit,'(I0,A)',advance='no') nsys_arg, ","
        write(unit,'(I0,A)',advance='no') nrow_min_arg, ","
        write(unit,'(I0,A)',advance='no') nrow_max_arg, ","
        write(unit,'(A)',advance='no') "device,"
        write(unit,'(ES24.16,A)',advance='no') global_sum_arg, ","
        write(unit,'(ES24.16,A)',advance='no') sqrt(global_sumsq_arg), ","
        write(unit,'(ES24.16,A)',advance='no') global_linf_arg, ","
        write(unit,'(ES24.16,A)',advance='no') global_samples_arg(1), ","
        write(unit,'(ES24.16,A)',advance='no') global_samples_arg(2), ","
        write(unit,'(ES24.16)') global_samples_arg(3)
    end subroutine write_signature_row

end program example_fortran_btdma_profile
