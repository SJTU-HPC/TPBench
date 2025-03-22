#!/bin/bash

source ./scripts/machine_config.sh 
 
result_dir=result/${machine}/comp_comm
mkdir -p $result_dir

function run_gemm_comm()
{
    np=$1
    mode=$2
    kernel=$3

    for size in 64 128 256 512
    do 
        log_path=${result_dir}/np${np}-${instr}-${mode}-${kernel}-N${size}.txt
        export TPBENCH_RUN_MODE=${instr}_${mode}
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -g ${kernel} -n 100 -s ${size}"
        echo $cmd
	    echo $log_path
        $cmd 2>&1 | tee ${log_path}
    done
}

function run_stencil_comm()
{
    np=$1
    mode=$2
    kernel=$3
    
    if [[ $mode == *"onesocket"* ]]; then
        map_by="core"
    else 
        map_by="socket"
    fi

    if [[ $mode == *"commonly"* ]]; then 
        export TPBENCH_SKIP_COMP=1
    else
        export TPBENCH_SKIP_COMP=0
    fi 

    for size in 200 500 1000 2000
    do

        if [[ $map_by == "core" ]]; then
            cmd="mpirun -np ${np} ${mpi_mca_conf} --bind-to core --map-by core ./tpbench.x -g ${kernel} -n 100 -s ${size}"
        else 
            # mpirun -np 64 ./scripts/numactl_socket.sh 64 ./tpbench.x -g d_jacobi2d5p_sendrecv -n 100 -s 200
            cmd="mpirun -np ${np} ${mpi_mca_conf} ./scripts/numactl_socket.sh ${np} ./tpbench.x -g ${kernel} -n 100 -s ${size}"
        fi
        log_path=${result_dir}/np${np}-${mode}-${kernel}-N${size}.txt
	    export TPBENCH_RUN_MODE=${mode}
        echo $cmd
	    echo $log_path
        $cmd 2>&1 | tee ${log_path}
    done
}

# run scalar gemm+comm
instr=scalar
make SETUP=ompi_${arch} clean
make SETUP=ompi_${arch}

for mode in compcomm commonly
do
    if [[ $mode == commonly ]]; then
        export TPBENCH_SKIP_COMP=1
    else
        export TPBENCH_SKIP_COMP=0
    fi
    for kernel in d_gemm_bcast d_gemm_allreduce
    do 
        run_gemm_comm $ncores $mode $kernel
    done
done

# run avx512 gemm+comm

if [[ $simd_mode != "none" ]]; then
    instr=${simd_mode}
    make SETUP=ompi_${arch}_${instr} clean
    make SETUP=ompi_${arch}_${instr} 
    for mode in compcomm commonly
    do
        if [[ $mode == commonly ]]; then
            export TPBENCH_SKIP_COMP=1
        else
            export TPBENCH_SKIP_COMP=0
        fi
        for kernel in d_gemm_bcast d_gemm_allreduce
        do 
            run_gemm_comm $ncores $mode $kernel
        done
    done
fi


# run stencil+comm

make SETUP=ompi_${arch} clean
make SETUP=ompi_${arch}
for mode in twosocket_compcomm twosocket_commonly onesocket_compcomm onesocket_commonly
do
    run_stencil_comm $ncores $mode d_jacobi2d5p_sendrecv
done


