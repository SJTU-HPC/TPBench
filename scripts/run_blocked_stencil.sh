#!/bin/bash

source ./scripts/machine_config.sh 

kernel_list="d_tl_cgw d_jacobi5p"
result_dir=result/${machine}/stencil
mkdir -p $result_dir

function run_array()
{
    np=$1
    kernel=$2
    log_path=${result_dir}/np${np}-${mode}-${kernel}.txt
    rm ${log_path}

    export ${block_env}
    for size in 100 1024 10240 
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 1000 -s ${size}"
        # $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done

    size=102400
    cmd="mpirun -np ${np} ${mpi_mca_conf}  --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 100 -s ${size}"
    # $cmd
    $cmd 2>&1 | tee -a ${log_path}

    size=1024000
    cmd="mpirun -np ${np} ${mpi_mca_conf}  --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 50 -s ${size}"
    # $cmd
    $cmd 2>&1 | tee -a ${log_path}
}

mode="noblock"
block_env=""
for kernel in ${kernel_list}
do 
    run_array 1 $kernel
    run_array $ncores $kernel
done

mode="block1000"
block_env="TPBENCH_BLOCK=1000"
for kernel in ${kernel_list}
do 
    run_array 1 $kernel
    run_array $ncores $kernel
done


