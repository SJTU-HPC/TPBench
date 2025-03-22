
machine="unknown"
ncores=1
arch=$(lscpu | grep "^Architecture:" | awk '{print $2}')
simd_mode="none"

if [[ $(arch) == *"x86"* ]]; then
    arch=x86
fi

cores_per_socket=$(lscpu | grep "^Core(s) per socket:" | awk '{print $4}')

if [[ $(hostname) == *"csprn1"* ]]; then
    machine="spr"
    simd_mode="avx512"
    ncores=56
elif [[ $(hostname) == *"amd9654"* ]]; then
    machine="amd9654"
    simd_mode="avx2"
    ncores=96
elif [[ $(hostname) == *"chgn1"* ]]; then
    machine="chgn1"
    simd_mode="avx2"
    ncores=64
    allreduce_alg_ids=(1 2 3 4 5 6)
    bcast_alg_ids=(1 2 3 4 5 6 7 8)
elif [[ $(hostname) == *"amd9754"* ]]; then
    if [[ "$cores_per_socket" = "128" ]]; then
        machine="amd9754"
    elif [[ "$cores_per_socket" = "96" ]]; then
        machine="amd9684X"
    else
        echo "Unknown machine"
        exit 1
    fi
    simd_mode="avx512"
    ncores=$cores_per_socket
    allreduce_alg_ids=(1 2 4 5 6)    
    bcast_alg_ids=(1 2 3 4 5 6 7)
elif [[ $(hostname) == *"amd9755"* ]]; then
    machine="amd9755"
    simd_mode="avx512"
    ncores=$cores_per_socket
elif [[  $(hostname) == *"920b"*  ]]; then
    machine="920b"
    arch=aarch64
    simd_mode=sve
    ncores=64
elif [[  $(hostname) == *"ft"*  ]]; then
    machine="ft"
    arch=aarch64
    simd_mode=neon
    mpi_mca_conf=" --mca pml ob1 --mca btl ^uct "
    ncores=$cores_per_socket
    export OMPI_MCA_pml=ob1
    export OMPI_MCA_btl=^uct
else
    echo "Unknown machine"
    exit 1
fi


# $arch $ncores $simd_mode 
# echo "machine: $machine"
# echo "arch: $arch"
# echo "ncores: $ncores"
# echo "simd_mode: $simd_mode"
# echo "allreduce_ids: (${allreduce_alg_ids[@]})"
# echo "bcast_ids: (${bcast_ids[@]})"
export machine arch ncores simd_mode
echo "CONFIG = {"
echo "    'machine': '$machine',"
echo "    'arch': '$arch'," 
echo "    'ncores': '$ncores',"
echo "    'simd_mode': '$simd_mode'"
echo "}"