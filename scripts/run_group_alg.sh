#!/bin/bash

# MPI_ALLREDUCE_ALG = ['fixed', 'basic_linear', 'nonoverlapping','recursive_doubling', 'ring', 'segmented_ring', 'rabenseifner']
# MPI_BCAST_ALG = ['fixed', 'basic_linear', 'chain', 'pipeline', 'split_binary_tree', 'binary_tree', 'binomial', 'knomial', 'scatter_allgather', 'scatter_allgather_ring']
allreduce_alg_ids=(0 1 2 3 4 5 6)
allreduce_algs=(fixed basicLinear nonoverlapping recursiveDoubling ring segmentedRing rabenseifner)
bcast_alg_ids=(0 1 2 3 4 5 6 7 8 9)
bcast_algs=(fixed basicLinear chain pipeline splitBinaryTree binaryTree binomial knomial scatterAllgather scatterAllgatherRing)

source ./scripts/machine_config.sh 
 
result_dir=result/${machine}/coll_alg
mkdir -p $result_dir

function run_gemm_comm()
{
    np=$1
    mode=$2
    kernel=$3
    alg=$4

    for size in 128 256 512 1024
    do 
        for Nr in 10 20 40
        do
            export TPBENCH_GEMM_NR=$Nr
            run_mode=${instr}_${mode}_${alg}
            log_path=${result_dir}/np${np}-${kernel}-${run_mode}-ntest100-N${size}-Nr${Nr}.txt
            export TPBENCH_RUN_MODE=${run_mode}
            cmd="mpirun -np ${np} --map-by core --bind-to core ./tpbench.x -g ${kernel} -n 100 -s ${size}"
            echo $cmd
            echo $log_path
            echo OMPI_MCA_coll_tuned_allreduce_algorithm=$OMPI_MCA_coll_tuned_allreduce_algorithm
            echo OMPI_MCA_coll_tuned_bcast_algorithm=$OMPI_MCA_coll_tuned_bcast_algorithm
            $cmd 2>&1 | tee ${log_path}
        done
    done
}

# run avx512 gemm+comm

instr=${simd_mode}
make SETUP=ompi_x86_${instr} clean
make SETUP=ompi_x86_${instr} 
for mode in commonly compcomm
do
    if [[ $mode == commonly ]]; then
        export TPBENCH_SKIP_COMP=1
        export TPBENCH_SKIP_COMM=0
    elif [[ $mode == compcomm ]]; then
        export TPBENCH_SKIP_COMP=0
        export TPBENCH_SKIP_COMM=0
    elif [[ $mode == componly ]]; then
        export TPBENCH_SKIP_COMP=0
        export TPBENCH_SKIP_COMM=1
    fi

    export OMPI_MCA_coll_tuned_use_dynamic_rules=1

    for alg in "${allreduce_alg_ids[@]}"
    do
        export OMPI_MCA_coll_tuned_allreduce_algorithm=$alg
        # echo "{allreduce_algs[$alg]}=${allreduce_algs[$alg]}"
        run_gemm_comm $ncores $mode d_gemm_allreduce ${allreduce_algs[$alg]}
    done
    export OMPI_MCA_coll_tuned_allreduce_algorithm=0
    
    for alg in "${bcast_alg_ids[@]}"
    do
        export OMPI_MCA_coll_tuned_bcast_algorithm=$alg
        # echo "{bcast_algs[$alg]}=${bcast_algs[$alg]}"
        run_gemm_comm $ncores $mode d_gemm_bcast ${bcast_algs[$alg]}
    done
    export OMPI_MCA_coll_tuned_bcast_algorithm=0
done


function run_gemm()
{
    np=$1
    size=$2
    mode=componly
    first_enter=1
    export TPBENCH_SKIP_COMP=0
    export TPBENCH_SKIP_COMM=1
    output_base=""
    log_base=""

    for kernel in d_gemm_allreduce d_gemm_bcast
    do 
        # Run only once time for Nr=10
        kernel_out=""
        export OMPI_MCA_coll_tuned_use_dynamic_rules=1
        mpi_alg_ids=()
        mpi_algs=()
        if [[ $kernel == "d_gemm_allreduce" ]]; then
            kernel_out="gemmallreduce"
            mpi_alg_env=OMPI_MCA_coll_tuned_allreduce_algorithm
            mpi_alg_ids=(${allreduce_alg_ids[@]})
            mpi_algs=(${allreduce_algs[@]})
        else
            kernel_out="gemmbcast"
            mpi_alg_env=OMPI_MCA_coll_tuned_bcast_algorithm
            mpi_alg_ids=(${bcast_alg_ids[@]})
            mpi_algs=(${bcast_algs[@]})
        fi
        for alg_id in "${mpi_alg_ids[@]}"
        do
            export $mpi_alg_env=$alg_id
            alg=${mpi_algs[$alg_id]}
            echo $alg_id
            echo $alg
            for Nr in 10 20 40
            do
                if [[ $first_enter == 1 ]]; then
                    first_enter=0
                    run_mode="${instr}_${mode}_${alg}"
                    output_base=${result_dir}/np${np}-${kernel}-${run_mode}-ntest100-N${size}-Nr${Nr}.txt
                    log_base1="result/log/np${np}-${kernel_out}-${run_mode}-GEMM(ns)-ntest100-N${size}-Nr${Nr}.csv"
                    log_base2="result/log/np${np}-${kernel_out}-${run_mode}-GEMM(cy)-ntest100-N${size}-Nr${Nr}.csv"
                    export TPBENCH_RUN_MODE=${run_mode}
                    export TPBENCH_GEMM_NR=$Nr
                    cmd="mpirun -np ${np} --map-by core --bind-to core ./tpbench.x -g ${kernel} -n 100 -s ${size}"
                    echo $cmd
                    echo $log_path
                    echo OMPI_MCA_coll_tuned_allreduce_algorithm=$OMPI_MCA_coll_tuned_allreduce_algorithm
                    echo OMPI_MCA_coll_tuned_bcast_algorithm=$OMPI_MCA_coll_tuned_bcast_algorithm
                    $cmd 2>&1 | tee ${output_base}
                else
                    run_mode="${instr}_${mode}_${alg}"
                    ouput_path=${result_dir}/np${np}-${kernel}-${run_mode}-ntest100-N${size}-Nr${Nr}.txt
                    log_path1="result/log/np${np}-${kernel_out}-${run_mode}-GEMM(ns)-ntest100-N${size}-Nr${Nr}.csv"
                    log_path2="result/log/np${np}-${kernel_out}-${run_mode}-GEMM(cy)-ntest100-N${size}-Nr${Nr}.csv"
                    cp $output_base $ouput_path
                    cp $log_base1 $log_path1
                    cp $log_base2 $log_path2
                fi
            done
        done
    done
}


for size in 128 256 512 1024
do
    run_gemm $ncores $size
done