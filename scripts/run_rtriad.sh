#!/bin/bash

function run_array()
{
    np=$1
    for size in 4 8 16 32 64 128 192 256 384 512 768 1024 1536 2048 3072 4096 8192 16384 32768 65536 102400 204800 409600 
    do
        mpirun -np ${np} --map-by core --bind-to core ./tpbench.x -k d_rtriad  -n 40 -s ${size}
    done
}


function run_mpi()
{
    for np in 1
    do
        run_array $np 2>&1 | tee -a result/rtriad/np${np}.log
    done
}

mkdir -p result/rtriad
run_mpi


