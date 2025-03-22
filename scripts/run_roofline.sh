#!/bin/bash

source ./scripts/machine_config.sh 

result_dir=result/${machine}/roofline
mkdir -p $result_dir

kernel_list="fmaldr mulldr"
size_list="16 32 512 2048 102400 1024000"
op_ratio_list="FL_RATIO_F1L4 FL_RATIO_F1L2 FL_RATIO_F1L1 FL_RATIO_F2L1 FL_RATIO_F4L1 FL_RATIO_F8L1"

function run() {
    np=$1
    kernel=$2

    for size in ${size_list}
    do
        log_path=${result_dir}/np${np}-${kernel}-${size}KB.txt
        rm ${log_path}

        for op_ratio in ${op_ratio_list}
        do 
            # first edit FL_RATIO_XX macro 
            file_path=./src/kernels/simple/${kernel}.c
            sed -i "s/^#define FL_RATIO.*$/#define ${op_ratio} 1/g" ${file_path}

            if [[ $simd_mode == "none" ]]; then
                make SETUP=ompi_${arch} clean
                make SETUP=ompi_${arch}
            else
                make SETUP=ompi_${arch}_${simd_mode} clean
                make SETUP=ompi_${arch}_${simd_mode}
            fi
            
            # then run benchmark
            cmd="mpirun -np ${np} ${mpi_mca_conf} --map-by core --bind-to core ./tpbench.x -k d_${kernel} -n 10 -s ${size}"
            echo $cmd
            $cmd 2>&1 | tee -a ${log_path}
        done
    done
}

echo "Note: before running this script. The number of uncommented '#define FL_RATIO_XX 1' lines in fmaldr.c and mulldr.c must be exactly 1"

for kernel in ${kernel_list}
do
    run 1 $kernel
done

for kernel in ${kernel_list}
do
    run $ncores $kernel
done