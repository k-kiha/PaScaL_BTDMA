program main
    use mpi
    use mod_btdma_cpu
    use mpiutil
    implicit none

    integer :: nprocs, myrank, ierr
    integer, parameter :: n1 = N1
    integer, parameter :: n2 = N2
    integer, parameter :: n3 = N3
    integer, parameter :: m  = M
    integer, parameter :: nrun = NRUN
    integer :: nsub, indxtmp_a, indxtmp_b;

    integer :: i,j,k
    integer :: n1sub,n2sub,n3sub

    real*8,allocatable,dimension(:,:,:,:,:) :: A,B,C
    real*8,allocatable,dimension(:  ,:,:,:) :: X,Xref

    type(BTDMA_PLAN) :: plan_ex

    integer :: run
    real*8 :: result_t0(1:nrun), result_t1(1:nrun), result_t2(1:nrun)
    real*8 :: result_t3(1:nrun), result_t4(1:nrun), result_t5(1:nrun)
    integer :: idx_min

    call MPI_INIT(ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, nprocs, ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)

    n1sub=n1
    n2sub=n2
    n3sub = mpiutil_para(1, n3, myrank, nprocs, indxtmp_a, indxtmp_b)

    allocate(A   (1:m,1:m,1:n1sub,1:n2sub,1:n3sub))
    allocate(B   (1:m,1:m,1:n1sub,1:n2sub,1:n3sub))
    allocate(C   (1:m,1:m,1:n1sub,1:n2sub,1:n3sub))
    allocate(X   (1:m,    1:n1sub,1:n2sub,1:n3sub))
    allocate(Xref(1:m,    1:n1sub,1:n2sub,1:n3sub))

    call btdma_makeplan(plan_ex,m,n1sub*n2sub,n3sub,MPI_COMM_WORLD)
    do run=1,nrun
        ! call demo_init(A,B,C,X,Xref,m,n1sub*n2sub,n3,n3sub,myrank,nprocs,indxtmp_a)
        A = 1.d0
        B = 2.d0
        C = 1.d0
        X = 1.d0
        Xref = 1.d0

        t0_a=0.d0;t0_b=0.d0
        t1_a=0.d0;t1_b=0.d0
        t2_a=0.d0;t2_b=0.d0
        t3_a=0.d0;t3_b=0.d0
        t4_a=0.d0;t4_b=0.d0
        t5_a=0.d0;t5_b=0.d0
        comm_tp_a=0.d0;comm_tp_b=0.d0
        comm_tc_a=0.d0;comm_tc_b=0.d0
        comm_tu_a=0.d0;comm_tu_b=0.d0

        call mpiutil_timecheck(t0_a,t0_b,0)

        call btdma_many_mpi(A,B,C,X,m,n1sub*n2sub,n3sub,plan_ex)
        ! call btdma_many_cyclic_mpi(A,B,C,X,m,n1sub*n2sub,n3sub,plan_ex)

        call mpiutil_timecheck(t0_a,t0_b,1)
        
        ! if(myrank==1) then        
        !     call viewV(X(1:m,i,j,n3sub),m)
        !     call viewV(Xref(1:m,i,j,n3sub),m)
        !     write(*,'(1A15,1I8)') "info. nprocs:",nprocs
        !     write(*,*) "..........................."

        !     write(*,*) "Total.           :", t0_b
        !     write(*,*) "btdma_modi_cpu   :", t1_b
        !     write(*,*) "a2av_forward_cpu :", t2_b
        !     write(*,*) "btdma_cpu        :", t3_b
        !     write(*,*) "a2av_backward_cpu:", t4_b
        !     write(*,*) "update_cpu       :", t5_b
        !     write(*,*) "  pack:",comm_tp_b
        !     write(*,*) "unpack:",comm_tu_b
        ! endif
        
        result_t0(run) = t0_b
        result_t1(run) = t1_b
        result_t2(run) = t2_b
        result_t3(run) = t3_b
        result_t4(run) = t4_b
        result_t5(run) = t5_b
    enddo
    call btdma_cleanplan(plan_ex)
    deallocate(A,B,C,X,Xref)

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
                ! A(i,j,sys,q) = 1.d0/(dble(i*10+j)*2.d0)*10.d0
                ! C(i,j,sys,q) = 1.d0/(dble(i*10+j)*3.d0)*10.d0
            end do
            end do
            B(1:m,1:m,sys,q) = II(1:m,1:m) -(A(1:m,1:m,sys,q)+C(1:m,1:m,sys,q))

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

        end do
    end do
end subroutine demo_init

subroutine viewM(A,m1,m2)
    integer :: m1,m2
    real*8  :: A(1:m1,1:m2)

    integer :: i,j

    write(*,*) '----------------------------------------'
    do i = 1, m1
        write(*,'(36F20.8)') (A(i,j),j=1,m2)
    end do
    write(*,*) '----------------------------------------'

end subroutine viewM

subroutine viewV(A,m1)
    integer :: m1
    real*8  :: A(1:m1)

    integer :: i

    write(*,*) '----------------------------------------'
    do i = 1, m1
        write(*,'(F30.16)') A(i)
    end do
    write(*,*) '----------------------------------------'

end subroutine viewV
