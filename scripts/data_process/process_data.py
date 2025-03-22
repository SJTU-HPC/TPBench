import os
import sys
import numpy as np
from collections import defaultdict

def parse_log(file_path, match_str='B/c', match_pos=4):
    rv = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            if line.startswith(match_str):
                # get median performance
                v = line.split()[match_pos]
                rv.append(v)
    return rv 


def process_blas1(input_dir: str, output_dir: str):

    def get_kernel_name(file_name):
        return file_name.split('-')[-1][:-4]
    
    file_list = os.listdir(input_dir)
    if len(file_list) == 0:
        return
    
    data = defaultdict(dict)

    kernels = set()
    for file in file_list:
        kernels.add(get_kernel_name(file))

    kernel_list = sorted(list(kernels))

    for file in file_list:
        tokens = file.split('-')
        key = tokens[0] + '-' + tokens[1]
        kernel_name = get_kernel_name(file)
        median_bpc = parse_log(os.path.join(input_dir, file))
        data[key][kernel_name] = median_bpc

    for key, value in data.items():
        output_path = os.path.join(output_dir, key + '.csv')
        f = open(output_path, 'w')
        for kernel in kernel_list:
            bpc_list = value[kernel]
            f.write(kernel + ',')
            f.write(','.join(bpc_list))
            f.write('\n')
        f.close()


def process_blocked_stencil(input_dir: str, output_dir: str):
    file_list = os.listdir(input_dir)
    if len(file_list) == 0:
        return
    
    data = {}

    for file in file_list:
        key = file[:-4]
        median_bpc = parse_log(os.path.join(input_dir, file))
        data[key] = median_bpc

    output_path = os.path.join(output_dir, 'stencil.csv')
    f = open(output_path, 'w')
    for key, value in sorted(data.items()):
        f.write(key + ',')
        f.write(','.join(value))
        f.write('\n')
    f.close()


def process_comp_comm(input_dir: str, output_dir: str):
    file_list = sorted(os.listdir(input_dir), key=lambda x: (int(x.split('-')[-1][1:-4]), x))
    if len(file_list) == 0:
        return
    
    kernel_list = ['gemm_bcast', 'gemm_allreduce', 'jacobi2d5p_sendrecv']
    for kernel in kernel_list:
        us_mat = []
        key_list = []
        for file in file_list:
            if file.find(kernel) == -1:
                continue
            tokens = file.split('-')
            key = '-'.join(tokens[:3]) + '-' + tokens[-1][:-4]
            key_list.append(key)
            median_us_list = parse_log(os.path.join(input_dir, file), match_str='comm(us)', match_pos=5)
            us_mat.append(median_us_list)
            
        us_mat = np.array(us_mat)
        us_mat = np.transpose(us_mat)            

        output_path = os.path.join(output_dir, f'{kernel}.csv')
        f = open(output_path, 'w')
        f.write(','.join(key_list) + '\n')
        for us_list in us_mat:
            f.write(','.join(us_list) + '\n')
        f.close()


def process_roofline(input_dir: str, output_dir: str):
    file_list = os.listdir(input_dir)
    if len(file_list) == 0:
        return
    
    kernel_list = ['fmaldr', 'mulldr']
    for kernel in kernel_list:
        data = {}
        for file in file_list:
            if file.find(kernel) == -1:
                continue
            tokens = file.split('-')
            key = tokens[0] + '-' + tokens[2][:-4]
            percentile95_list = parse_log(os.path.join(input_dir, file), match_str='B/c', match_pos=6)
            data[key] = percentile95_list

        output_path = os.path.join(output_dir, f'{kernel}.csv')
        f = open(output_path, 'w')
        for key, value in sorted(data.items(), key=lambda x: (int(x[0].split('-')[0][2:]), int(x[0].split('-')[-1][:-2]))):
            f.write(key + ',')
            f.write(','.join(value))
            f.write('\n')
        f.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python <this.py> <device_result_dir>. e.g.: "python scripts/process_data.py result/amd9654"')
    result_dir = sys.argv[1]
    output_dir = os.path.join(result_dir, 'summary')
    os.makedirs(output_dir, exist_ok=True)

    # blas1

    blas1_input_dir = os.path.join(result_dir, 'blas1')
    blas1_output_dir = os.path.join(output_dir, 'blas1')
    os.makedirs(blas1_output_dir, exist_ok=True)
    process_blas1(blas1_input_dir, blas1_output_dir)

    # blocked_stencil

    stencil_input_dir = os.path.join(result_dir, 'stencil')
    stencil_output_dir = os.path.join(output_dir, 'stencil')
    os.makedirs(stencil_output_dir, exist_ok=True)
    process_blocked_stencil(stencil_input_dir, stencil_output_dir)

    # comp+comm

    comm_input_dir = os.path.join(result_dir, 'comp_comm')
    comm_output_dir = os.path.join(output_dir, 'comp_comm')
    os.makedirs(comm_output_dir, exist_ok=True)
    process_comp_comm(comm_input_dir, comm_output_dir)

    # roofline

    input_dir = os.path.join(result_dir, 'roofline')
    output_dir = os.path.join(output_dir, 'roofline')
    os.makedirs(output_dir, exist_ok=True)
    process_roofline(input_dir, output_dir)

