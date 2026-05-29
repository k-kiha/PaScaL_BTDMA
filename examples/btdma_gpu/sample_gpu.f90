program main
    use mod_btdma_gpu_v2
    use cudafor
    use mpiutil
    use mpi
    implicit none

    integer :: nprocs, myrank, ierr
    integer :: nsub, indxtmp_a, indxtmp_b;
    
    integer, parameter :: n1 = N1
    integer, parameter :: n2 = N2
    integer, parameter :: n3 = N3
    integer, parameter :: m  = M
    integer, parameter :: nrun = NRUN
    
    integer :: i,j,k,rank
    integer :: n1sub,n2sub,n3sub

    real*8,allocatable,dimension(:,:,:,:,:) :: A,B,C
    real*8,allocatable,dimension(:  ,:,:,:) :: X,Xref

    real*8,allocatable,dimension(:,:,:,:,:) :: MMA,NNA
    real*8,allocatable,dimension(:,:,:,:,:) :: MMC,NNC


    real*8, device, allocatable, dimension(:,:,:,:,:) :: d_A2,d_B2,d_C2
    real*8, device, allocatable, dimension(:,:,:,:  ) :: d_X2
    
    type(dim3) :: threads, blocks

    integer :: dev, ngpu, gpurank
    type(cudadeviceprop) :: prop

    integer :: run

    real*8 :: result_t0(1:nrun), result_t1(1:nrun), result_t2(1:nrun)
    real*8 :: result_t3(1:nrun), result_t4(1:nrun), result_t5(1:nrun)
    integer :: idx_min

    type(BTDMA_PLAN_gpu_v2) :: plan_ex_v2

    real*8 :: time_v0, time_v1, time_v2

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
    
    ! Assign GPU
    ierr = cudaGetDeviceCount(ngpu)
    gpurank = mod(myrank, ngpu)
    ierr = cudaSetDevice(gpurank)
    ierr = cudaDeviceSynchronize()
    ! ierr = cudaDeviceSetLimit(cudaLimitMallocHeapSize, 512_8 * 1024_8 * 1024_8)
    ierr = cudaGetDevice(dev)
    ierr = cudaGetDeviceProperties( prop, dev )

    n1sub=n1
    n2sub=n2
    n3sub = mpiutil_para(1, n3, myrank, nprocs, indxtmp_a, indxtmp_b)

    allocate(d_A2(1:n1sub,1:n2sub,1:n3sub,1:m,1:m))
    allocate(d_B2(1:n1sub,1:n2sub,1:n3sub,1:m,1:m))
    allocate(d_C2(1:n1sub,1:n2sub,1:n3sub,1:m,1:m))
    allocate(d_X2(1:n1sub,1:n2sub,1:n3sub,1:m    ))

    if(myrank==0) then        
        write(*,'(1A30,1A1)',advance='no') trim(prop%name),"|"
        write(*,'(1A8,1A1)',advance='no') "info. nprocs","|"
        write(*,'(1A8,1A1)',advance='no') "n1","|"
        write(*,'(1A8,1A1)',advance='no') "n2","|"
        write(*,'(1A8,1A1)',advance='no') "n3","|"
        write(*,'(1A8,1A1)',advance='no') "m","|"
        write(*,'(1A25,1A1)',advance='no') "Total","|"
        write(*,'(1A25,1A1)',advance='no') "btdma_modi_gpu","|"
        write(*,'(1A25,1A1)',advance='no') "a2av_forward_gpu","|"
        write(*,'(1A25,1A1)',advance='no') "btdma_gpu","|"
        write(*,'(1A25,1A1)',advance='no') "a2av_backward_gpu","|"
        write(*,'(1A25,1A1)',advance='no') "update_gpu","|"
        write(*,*)
    endif

    call btdma_makeplan_gpu_v2(plan_ex_v2,m,n1sub*n2sub,n3sub,MPI_COMM_WORLD)
    do run=1,nrun
        
        d_A2 = 1.0
        d_B2 = 2.0
        d_C2 = 1.0
        d_X2 = 1.0

        t0__a=0.d0;t0__b=0.d0
        t1__a=0.d0;t1__b=0.d0
        t2__a=0.d0;t2__b=0.d0
        t3__a=0.d0;t3__b=0.d0
        t4__a=0.d0;t4__b=0.d0
        t5__a=0.d0;t5__b=0.d0

        call MPI_BARRIER(MPI_COMM_WORLD,ierr)
        call btdma_timecheck(t0__a,t0__b,0)

        call btdma_many_mpi_gpu_v2(d_A2,d_B2,d_C2,d_X2,m,n1sub*n2sub,n3sub,plan_ex_v2)
        ! call btdma_many_cycl_mpi_gpu_v2(d_A2,d_B2,d_C2,d_X2,m,n1sub*n2sub,n3sub,plan_ex_v2)

        call MPI_BARRIER(MPI_COMM_WORLD,ierr)
        call btdma_timecheck(t0__a,t0__b,1)

        result_t0(run) = t0__b
        result_t1(run) = t1__b
        result_t2(run) = t2__b
        result_t3(run) = t3__b
        result_t4(run) = t4__b
        result_t5(run) = t5__b

    enddo

    idx_min = minloc(result_t0(1:nrun), dim=1)
    if(myrank==0) then        
        write(*,'(1A30,1A1)',advance='no') " ","|"
        write(*,'(1I8,1A1)',advance='no') nprocs,"|"
        write(*,'(1I8,1A1)',advance='no') n1,"|"
        write(*,'(1I8,1A1)',advance='no') n2,"|"
        write(*,'(1I8,1A1)',advance='no') n3,"|"
        write(*,'(1I8,1A1)',advance='no') m,"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t0(idx_min),"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t1(idx_min),"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t2(idx_min),"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t3(idx_min),"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t4(idx_min),"|"
        write(*,'(1E25.15,1A1)',advance='no') result_t5(idx_min),"|"
        write(*,*)
    endif

    call btdma_cleanplan_gpu_v2(plan_ex_v2) 

    deallocate(d_A2,d_B2,d_C2,d_X2)
    call MPI_FINALIZE(ierr)
end program main

subroutine demo_init(A,B,C,X,Xref,m,nsys,nall,nsub,myrank,nprocs,indxtmp_a)
    integer :: m,nall,nsys,nsub,myrank,nprocs,indxtmp_a
    real*8  :: A(1:m,1:m   ,1:nsys,1:nsub) 
    real*8  :: B(1:m,1:m   ,1:nsys,1:nsub) 
    real*8  :: C(1:m,1:m   ,1:nsys,1:nsub) 
    real*8  :: X(1:m       ,1:nsys,1:nsub) 
    real*8  :: Xref(1:m    ,1:nsys,1:nsub) 

    integer :: i,j,sys,q
    real*8  :: II(1:m,1:m)
    real*8  :: Vtmp(1:m),VtmpU(1:m),VtmpL(1:m)
    real*8  :: VtmpU0(1:m),VtmpL0(1:m)
    
    call random_seed()

    II(1:m,1:m) = 0.d0
    do i = 1, m
        II(i,i) = 1.d0
    enddo

    do q = 1, nsub
        do sys = 1, nsys
            do i = 1,m
                Xref(i,sys,q) = dble((indxtmp_a-1+q)*100+i)
            end do

            do j = 1,m
            do i = 1,m
                call random_number(A(i,j,sys,q))
                call random_number(C(i,j,sys,q))
                A(i,j,sys,q) = (A(i,j,sys,q)-0.5d0) * 0.8d0
                C(i,j,sys,q) = (C(i,j,sys,q)-0.5d0) * 0.8d0
            end do
            end do
            B(1:m,1:m,sys,q) = II(1:m,1:m) -(A(1:m,1:m,sys,q)+C(1:m,1:m,sys,q))

            ! !----check
            ! B(1:m,1:m,sys,q) = 0.d0
            ! do j = 1,m
            !     B(j,j,sys,q) = -2.d0
            !     if(j<m) B(j,j+1,sys,q) = 1.d0
            !     if(j>1) B(j,j-1,sys,q) = 1.d0
            ! enddo
        end do
    end do
    
    do q = 1, nsub
        do sys = 1, nsys
            do i = 1,m
                Vtmp (i) = dble((indxtmp_a-1+q  )*100+i)
                VtmpU(i) = dble((indxtmp_a-1+q+1)*100+i)
                VtmpL(i) = dble((indxtmp_a-1+q-1)*100+i)
                VtmpU0(i) = dble((1   )*100+i)
                VtmpL0(i) = dble((nall)*100+i)
            end do

            X(1:m,sys,q) = matmul(B(1:m,1:m,sys,q),Vtmp (1:m)) &
                         + matmul(A(1:m,1:m,sys,q),VtmpL(1:m)) &
                         + matmul(C(1:m,1:m,sys,q),VtmpU(1:m))
            if(myrank==0       .and.q==1   ) X(1:m,sys,q) = X(1:m,sys,q) - matmul(A(1:m,1:m,sys,q),VtmpL(1:m))
            if(myrank==nprocs-1.and.q==nsub) X(1:m,sys,q) = X(1:m,sys,q) - matmul(C(1:m,1:m,sys,q),VtmpU(1:m))
            ! if(myrank==0       .and.q==1   ) X(1:m,sys,q) = X(1:m,sys,q) - matmul(A(1:m,1:m,sys,q),VtmpL(1:m)) + matmul(A(1:m,1:m,sys,q),VtmpL0(1:m))
            ! if(myrank==nprocs-1.and.q==nsub) X(1:m,sys,q) = X(1:m,sys,q) - matmul(C(1:m,1:m,sys,q),VtmpU(1:m)) + matmul(C(1:m,1:m,sys,q),VtmpU0(1:m))

            ! !----check
            ! C(1:m,1:m,sys,q) = 0.d0
            ! do j = 1,m
            !     C(1,j,sys,q) = -j
            !     C(m,j,sys,q) = -j
            ! enddo
            ! X(1:m,sys,q) = 0.d0
            ! X(1,sys,q) =-10.d0
            ! X(m,sys,q) =-10.d0
        end do
    end do
end subroutine demo_init

subroutine viewM(A,m1,m2)
    integer :: m1,m2
    real*8  :: A(1:m1,1:m2)

    integer :: i,j

    do i = 1, m1
        write(*,'(36F20.8)') (A(i,j),j=1,m2)
    end do
    write(*,*) '----------------------------------------'

end subroutine viewM

subroutine viewV(A,m1,flag)
    integer :: m1, flag
    real*8  :: A(1:m1)

    integer :: i

    do i = 1, m1
        if (flag==0) then
            write(*,'(A,F30.16)') "** ",A(i)
        else
            write(*,'(A,F30.16)') ">> ",A(i)
        endif
    end do
    write(*,*) '----------------------------------------'

end subroutine viewV

