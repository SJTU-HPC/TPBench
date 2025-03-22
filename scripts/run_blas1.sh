#!/bin/bash

source ./scripts/machine_config.sh 

kernel_list="d_init d_sum d_copy d_update d_triad d_striad d_axpy d_scale d_staxpy"
result_dir=result/${machine}/blas1
mkdir -p $result_dir

function run_array_np1()
{
    np=1
    kernel=$1
    log_path=${result_dir}/np${np}-${mode}-${kernel}.txt
    rm ${log_path}

    for size in 4 8 16 32 64 128 256 384 512 768 1024
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 1000 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
    for size in 1536 2048 3072 4096 8192 16384
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 300 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
    for size in 32768 65536 102400 204800 409600 
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 100 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
}

function run_array_np()
{
    np=$1
    kernel=$2

    log_path=${result_dir}/np${np}-${mode}-${kernel}.txt
    rm ${log_path}

    for size in 4 8 16 32 64 128 256 384 512 768 1024
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 1000 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
    for size in 1536 2048 3072 4096 
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 300 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
    for size in 8192 16384 32768 65536 102400 204800 409600 
    do
        cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k ${kernel}  -n 100 -s ${size}"
        echo $cmd
        $cmd 2>&1 | tee -a ${log_path}
    done
}

mode="scalar"
make SETUP=ompi_${arch} clean
make SETUP=ompi_${arch}

for kernel in ${kernel_list}
do
    run_array_np1 $kernel
done

for kernel in ${kernel_list}
do
    run_array_np $ncores $kernel
done


if [[ $simd_mode != "none" ]]; then
    mode=${simd_mode}
    make SETUP=ompi_${arch}_${mode} clean
    make SETUP=ompi_${arch}_${mode}

    for kernel in ${kernel_list}
    do
        run_array_np1 $kernel
    done

    for kernel in ${kernel_list}
    do
        run_array_np $ncores $kernel
    done
fi
