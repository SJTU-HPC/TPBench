#!/bin/bash

ompi_rank=$OMPI_COMM_WORLD_RANK
socket_size=$1
cmd="${@:2}"

half_socket_size=$(($socket_size/2))
offset=$(($ompi_rank/$half_socket_size))

core_id=$(($offset*$socket_size + $ompi_rank % $half_socket_size))

# echo "numactl --all -C $core_id -l $cmd"
numactl --all -C $core_id -l $cmd