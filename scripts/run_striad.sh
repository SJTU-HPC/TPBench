
function run_one()
{
    stride=$1
    l=$1

    mpirun -np 1 --map-by core --bind-to core -x TPBENCH_STRIDE=${stride} -x TPBENCH_L=${l} ./tpbench.x -k d_striad  -n 300 -s 40960
}

mkdir -p result/striad

for x in 8 16 32 64 128 256 512 1024 2048
do 
    run_one $x $x 2>&1 | tee -a result/striad/np1.log
done