'''
This script is used to process the data of the collective communication algorithms.
We classify the results into three levels:
- task: A task is the minimum unit of program execution. It includes the data of 
        a single kernel with a fixed run_mode parameter execution
- kernel: A kernel is a collection of tasks with the same kernel_name, processes(np)
            , ntest, N, Nr and so on. A kernel can be executed with different 
            run_mode parameters, which generates different tasks.
- program_chunk: A program_chunk is one of the execution part of a task, which can 
                    be a computation part or a communication part with different 
                    units. The program_chunk is used to distinguish the different 
                    parts of the task.

There are some key task parameters which be classified into two categories:
- Problem Parameters:
    - np: The number of processes
    - kernel_name: The name of the TPBench's kernel
- Environment Parameters:
    - run_mode: A collection of all execution parameters
        - compute_unit: The compute unit of the CPU, such as avx2, avx512, scalar
        - program_pattern: The programs' pattern, such as compcomm, commonly, componly
        - socket: The number of sockets, such as onesocket, twosocket
        - mpi_alg: The MPI coll func's algorithm. "fix" is the default algorithm.

According to the different levels of data, we provide different classed:
- Tasks: A class to store all the tasks' data
- Kernel_datas: A class to store the data of a single kernel
- Task_Datas: A class to store the data of a single task
- Chunk_Data: A class to store the data of a single program_chunk

And for now, the connection between these classes is:
- Tasks: A collection of Kernel_datas and Task_Datas 
- Kernel_datas: A collection of Task_Datas's statistics datas
- Task_Datas: A collection of Chunk_Data


In the future, the connection between these classes will be:
- Tasks: A collection of Kernel_datas
    - Compare the different kernels' performance:
        - Aspect1: different np, ntest, N, Nr
    - Analyze the performance from Kernels' level
- Kernel_datas: A collection of Task_Datas with different run_mode
    - Compare the different tasks' performance through run_mode's aspects
    - Summary the different program_chunks' performance
    - Analyze Task_Datas' total wall time data
- Task_Datas: A collection of Chunk_Data
    - Summary the different program_chunks' performance
    - Analyze program_chunks' statistics data
- Chunk_Data: A collection of raw data
    - Process the raw data and store the processed data
    - Analyze the processed raw data
'''


import pandas as pd
import numpy as np
import os
import re
import socket
import tp_coll_plot as TPcp



class Run_Modes:
    MPI_ALLREDUCE_ALG = ['fixed', 'basicLinear', 'nonoverlapping', 'recursiveDoubling', 'ring', 'segmentedRing', 'rabenseifner']
    MPI_BCAST_ALG = ['fixed', 'basicLinear', 'chain', 'pipeline', 'splitBinaryTree', 'binaryTree', 'binomial', 'knomial', 'scatterAllgather', 'scatterAllgatherRing']

    def __init__(self, run_mode:str):
        self.run_mode = run_mode
        self.program_pattern = ''
        self.compute_unit = ''
        self.socket = ''
        self.mpi_alg = ''
        self.__extract_run_mode()
        
    def __extract_run_mode(self):
        if 'compcomm' in self.run_mode:
            self.program_pattern = 'compcomm'
        elif 'commonly' in self.run_mode:
            self.program_pattern = 'commonly'
        elif 'componly' in self.run_mode:
            self.program_pattern = 'componly'
        
        if 'twosocket' in self.run_mode:
            self.socket = 'twosocket'
        elif 'onesocket' in self.run_mode:
            self.socket = 'onesocket'
        if 'avx2' in self.run_mode:
            self.compute_unit = 'avx2'
        elif 'avx512' in self.run_mode:
            self.compute_unit = 'avx512'
        elif 'scalar' in self.run_mode:
            self.compute_unit = 'scalar'
        
        if len(self.run_mode.split('_')) == 3:
            if self.run_mode.split('_')[2] in Run_Modes.MPI_ALLREDUCE_ALG or self.run_mode.split('_')[2] in Run_Modes.MPI_BCAST_ALG:
                self.mpi_alg = self.run_mode.split('_')[2]
            
    def get_run_mode_sub(self):
        return self.compute_unit, self.program_pattern, self.socket, self.mpi_alg

    def get_GEMM_run_modes(compute_units: list[str], program_patterns:list[str]):
        run_modes = []
        for cu in compute_units:
            for pp in program_patterns:
                if cu != '':
                    cu += '_'
                run_mode = f'{cu}{pp}'
                run_modes.append(run_mode)
        return run_modes

    def get_Jacobi_run_modes(sockets: list[str], program_patterns:list[str]):
        run_modes = []
        for s in sockets:
            for pp in program_patterns:
                if s != '':
                    s += '_'
                run_mode = f'{s}{pp}'
                run_modes.append(run_mode)
        return run_modes

    def get_algorithm_run_modes(compute_units: list[str], program_patterns:list[str], mpi_funcs:str, mpi_func_algs: dict[str, list[str]]):
        run_modes = []
        for cu in compute_units:
            if cu != '':
                cu += '_'
            for pp in program_patterns:
                if pp != '':
                    pp += '_'
                for mf in mpi_funcs:
                    if mf in mpi_func_algs:
                        for alg in mpi_func_algs[mf]:
                            run_mode = f'{cu}{pp}{alg}'
                            run_modes.append(run_mode)
                    else:
                        exit(f"Invalid mpi function = {mf}.")
        return run_modes

    def generate_run_modes(mode: int, program_patterns: list[str], compute_units: list[str], sockets:list[str], mpi_funcs:str, mpi_func_algs: dict[str, list[str]]):
        run_modes = []
        if mode == 0:
            run_modes = Run_Modes.get_GEMM_run_modes(compute_units, program_patterns)
        elif mode == 1:
            run_modes = Run_Modes.get_Jacobi_run_modes(sockets, program_patterns)
        elif mode == 2:
            run_modes = Run_Modes.get_algorithm_run_modes(compute_units, program_patterns, mpi_funcs, mpi_func_algs)
        else:
            exit(f"Invalid mode = {mode}.")
        return run_modes

class Chunk_Data:
    time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    hostname = socket.gethostname()
    machine = None
    if "csprn1" in hostname:
        machine = "spr"
    elif "amd9654" in hostname:
        machine = "amd9654"
    elif "chgn1" in hostname:
        machine = "chgn1"
    elif "amd9754" in hostname:
        machine = "amd9754"
    if machine is None:
        exit("Invalid hostname.")

    def update_time_stamp():
        Chunk_Data.time_stamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    def get_latest_time_stamp(result_dir:str):
        local_dirs = []
        for d in os.listdir(result_dir):
            if os.path.isdir(os.path.join(result_dir, d)):
                if d.startswith(f'{Chunk_Data.machine}-') and len(d.split('-')) == 4:
                    if d.split('-')[1].startswith('20'):
                        local_dirs.append(d)
        if len(local_dirs) > 0:
            latest_dir = sorted(local_dirs, key=lambda x: x.split('-')[1], reverse=True)[0]
            Chunk_Data.time_stamp = latest_dir.split('-')[1]

    def __init__(self, task_name:str, total_time:float, program_chunk:str, raw_unit:str, raw_data:pd.DataFrame):
        self.program_chunk = program_chunk
        self.np, self.tpkernel_name, self.run_mode, self.ntest, self.N, self.Nr = self._extract_task_name(task_name)
        self.total_time_per_step = total_time / self.ntest
        self.run_mode_sub = Run_Modes(self.run_mode)
        self.raw_data = raw_data
        self.raw_unit = raw_unit
        self.data_size_operate = 1              # Real size of data in KiB 
        self.data_size_memory = 1               # Memory access size of in KiB 
        self.processed_data:dict[str,pd.DataFrame] = {}

    def _extract_task_name(self, task_name):
        # file_name = np{np}-{tpkernel_name}-{run_mode}-{program_chunk}({counter_unit})-ntest{ntest}-N{N}.csv
        # task_name = np{np}-{tpkernel_name}-{run_mode}-ntest{ntest}-N{N}
        tpkernel_name = task_name.split("-")[1]
        run_mode = task_name.split("-")[2]
        np = int(re.search(r'np(\d+)', task_name).group(1))
        ntest = int(re.search(r'-ntest(\d+)-', task_name).group(1))
        N = int(re.search(r'-N(\d+)', task_name).group(1))
        Nr = int(re.search(r'-Nr(\d+)', task_name).group(1))
        return np, tpkernel_name, run_mode, ntest, N, Nr

    def process_data(self, processed_unit, mode=0):
        if self.raw_data is not None:
            # Convert ns to us
            if 'comm' in self.program_chunk:
                if self.raw_unit == 'ns': 
                    data = 1 
                    if 'gemm' in self.tpkernel_name:
                        data = self.Nr * self.N * 8
                        self.data_size_memory = 8 * 3 * self.N ** 2 / 1024
                    elif 'jacobi' in self.tpkernel_name:
                        data = self.N * 16
                        self.data_size_memory = 8 * 2 * self.N ** 2 / 1024
                    # else:
                    #     exit("Invalid kernel name.")

                    self.data_size_operate = data/1024 if self.data_size_operate == 1 else self.data_size_operate
                    if processed_unit == 'us':
                        self.processed_data[processed_unit] = self.raw_data.astype(float) / 1000
                    elif processed_unit == 'KB_per_sec':
                        self.processed_data[processed_unit] = data * 1000000 / self.raw_data.astype(float)
            # Convert ns to GFLOPs
            elif 'GEMM' in self.program_chunk or 'Jacobi' in self.program_chunk:
                flop = 1.0
                if 'GEMM' in self.program_chunk :
                    flop = 2.0 * self.N ** 3
                    self.data_size_memory = 8 * 3 * self.N ** 2 / 1024
                elif 'Jacobi' in self.program_chunk:
                    flop = 3.0 * self.N ** 3
                    self.data_size_memory = 8 * 2 * self.N ** 2 / 1024

                self.data_size_operate = flop/1024 if self.data_size_operate == 1 else self.data_size_operate
                if self.raw_unit == 'ns': 
                    if processed_unit == 'GFLOPs':
                        self.processed_data[processed_unit] = flop / self.raw_data.astype(float)
                    elif processed_unit == 'KB_per_sec':
                        self.processed_data[processed_unit] = 8*flop / self.raw_data.astype(float)
                    elif processed_unit == 'us':
                        self.processed_data[processed_unit] = self.raw_data.astype(float) / 1000
                elif self.raw_unit == 'cy':
                    if processed_unit == 'bytes_per_cycle':
                        self.processed_data[processed_unit] = 8*flop/self.raw_data.astype(float)
            else:
                exit("Invalid program chunk.")
        else:
            print("Data not loaded.")

    def draw_interactive_figures(self, processed_units=[], save_dir=''):
        if processed_units == []:
            processed_units = self.processed_data.keys()

        for processed_unit in processed_units:
            if processed_unit not in self.processed_data:
                print(f"{processed_unit} not found.")
                continue
            processed_data = self.processed_data[processed_unit]
            title_name = f'{self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode}<br>{processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest} total_time/step={self.total_time_per_step:.4g}us'
            file_name = f'{self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode} {processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest}'.replace(' ', '-')
            directory = os.path.join(save_dir, f'{self.machine}-{self.time_stamp}-result-interactive')
            TPcp.draw_interactive_figures(data_to_plot = processed_data, title_name = title_name, save_dir = directory, file_name = file_name)


    def draw_fixed_heatmap(self, processed_units=[], save_dir=''):
        if processed_units == []:
            processed_units = self.processed_data.keys()
        for processed_unit in processed_units:
            if processed_unit not in self.processed_data:
                print(f"{processed_unit} not found.")
                continue
            processed_data = self.processed_data[processed_unit]
            title_name = f'Heatmap of {self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode}\n{processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest} total_time/step={self.total_time_per_step:.4g}us'
            file_name = f'Heatmap of {self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode} {processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest}'.replace(' ', '-')
            directory = os.path.join(save_dir, f'{self.machine}-{self.time_stamp}-result-heatmap')
            TPcp.draw_fixed_heatmap(data_to_plot = processed_data, title_name = title_name, save_dir = directory, file_name = file_name)
            

    def draw_fixed_histogram(self, processed_units=[], mode=0, save_dir=''):
        if processed_units == []:
            processed_units = self.processed_data.keys()
        for processed_unit in processed_units:
            if processed_unit not in self.processed_data:
                print(f"{processed_unit} not found.")
                continue
            processed_data = self.processed_data[processed_unit]
            title_name = f'Histogram of {self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode}\n{processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest} total_time/step={self.total_time_per_step:.4g}us'
            file_name = f'Histogram of {self.tpkernel_name} {self.program_chunk.split("(")[0]} {self.run_mode} {processed_unit} with np={self.np} N={self.N} Nr={self.Nr} ntest={self.ntest}'.replace(' ', '-')
            dictionary = os.path.join(save_dir, f'{self.machine}-{self.time_stamp}-result-histogram')
            if mode == 0:
                title_name += '\n Steps\' Analysis Result'
                file_name += ' Steps\' Analysis Result'.replace(' ', '-')
                data_to_plot = processed_data.describe().loc[['mean', 'min', 'max', '50%']].T
                TPcp.draw_2D_histogram(data_to_plot = data_to_plot, title_name = title_name, file_name = file_name, save_dir = dictionary)
            elif mode == 1:
                title_name += '\n Ranks\' Analysis Result'
                file_name += ' Ranks\' Analysis Result'.replace(' ', '-')
                data_to_plot = processed_data.T.describe().loc[['mean', 'min', 'max', '50%']].T
                TPcp.draw_2D_histogram(data_to_plot = data_to_plot, title_name = title_name, file_name = file_name,  save_dir = dictionary)
            elif mode == 2:
                data_to_plot = pd.DataFrame(processed_data.values.flatten(), columns=['values'])
                TPcp.draw_1D_histogram(data_to_plot = data_to_plot, title_name = title_name, file_name = file_name, save_dir = dictionary)


class Task_Datas:
    '''
    Task_Datas class is used to store all the data of A TASK.
    '''

    def __init__(self, file_path, task_name, program_chunk, raw_unit, raw_data):
        self.task_name = task_name
        self.output_path = os.path.join(file_path, Chunk_Data.machine, 'coll_alg')
        self.total_time:float=0 #us
        self.get_total_time()
        self.datas:dict[str,Chunk_Data] = {}  # Dictionary to hold Chunk_Data instances
        self.add_data(program_chunk, raw_unit, raw_data)

    def get_total_time(self):
        tpkernel_name_origin=self.task_name.split('-')[1]
        tpkernel_name=tpkernel_name_origin.replace('gemm', 'd_gemm_')
        output_file_name = self.task_name.replace(tpkernel_name_origin, tpkernel_name) + '.txt'
        output_file_path = os.path.join(self.output_path, output_file_name)
        if os.path.exists(output_file_path):
            with open(output_file_path, 'r') as f:
                for line in f:
                    if 'Total Wall Time(us):' in line:
                        self.total_time = float(line.split()[-1])
        else:
            print(f"{output_file_path} not found.")

    def _procss_data(self, program_chunk, raw_unit):
        if program_chunk not in self.datas:
            exit("Program chunk not found.")
        processed_unit = None
        if 'comm' in program_chunk:
            mode = 0
            processed_unit = 'us'
            self.datas[program_chunk].process_data(processed_unit, mode)
            processed_unit = 'KB_per_sec'
            self.datas[program_chunk].process_data(processed_unit, mode)
        elif 'GEMM' in program_chunk or 'Jacobi' in program_chunk:
            mode = 1
            if raw_unit == 'cy':
                # processed_unit = 'bytes_per_cycle'
                # self.datas[program_chunk].process_data(processed_unit, mode)
                pass
            elif raw_unit == 'ns':
                processed_unit = 'us'
                self.datas[program_chunk].process_data(processed_unit, mode)
                processed_unit = 'KB_per_sec'
                self.datas[program_chunk].process_data(processed_unit, mode)
                # processed_unit = 'GFLOPs'
                # self.datas[program_chunk].process_data(processed_unit, mode)
        else:
            exit("Invalid program chunk.")

    def add_data(self, program_chunk, raw_unit, raw_data):
        if program_chunk not in self.datas:
            self.datas[program_chunk] = Chunk_Data(self.task_name, self.total_time, program_chunk, raw_unit, raw_data)
        else:
            exit("Program chunk already exists.")
        self._procss_data(program_chunk, raw_unit)


    def draw_figures(self, program_chunks = [], processed_units_dict:dict[str,list] = {}, figure_mode='fixed', save_dir = ''):
        if program_chunks == []:
            program_chunks = self.datas.keys()
        for program_chunk in program_chunks:
            if program_chunk in self.datas:
                processed_units = []
                if program_chunk in processed_units_dict:
                    processed_units = processed_units_dict[program_chunk]
                if figure_mode == 'fixed':
                    self.datas[program_chunk].draw_fixed_heatmap(processed_units = processed_units, save_dir=save_dir)
                    self.datas[program_chunk].draw_fixed_histogram(processed_units = processed_units, mode=2, save_dir=save_dir)
                elif figure_mode == 'interactive':
                    self.datas[program_chunk].draw_interactive_figures(processed_units = processed_units, save_dir=save_dir)
                else:
                    exit(f"Invalid figure mode = {figure_mode}.")

    def draw_all_chunks(self, program_chunks:list, processed_units_dict:dict, result_path:str):
        # Draw all program_chunks' figures in a single figure
        data_chunks = []
        for program_chunk in program_chunks:
            if program_chunk in self.datas:
                data_chunks.append(program_chunk)

        if len(data_chunks) == 0:
            exit(f"Invalid number of program chunks = {data_chunks}.")

        machine = Chunk_Data.machine
        local_histogram_dirs = []
        local_heatmap_dirs = []
        for d in os.listdir(result_path):
            if os.path.isdir(os.path.join(result_path, d)):
                if d.startswith(f'{machine}-') and d.endswith('result-histogram'):
                    local_histogram_dirs.append(d)
                elif d.startswith(f'{machine}-') and d.endswith('result-heatmap'):
                    local_heatmap_dirs.append(d)
        histogram_dir = sorted(local_histogram_dirs, key=lambda x: x.split('-')[1], reverse=True)[0]
        heatmap_dir = sorted(local_heatmap_dirs, key=lambda x: x.split('-')[1], reverse=True)[0]

        heatmap_dir = os.path.join(result_path, heatmap_dir)
        histogram_dir = os.path.join(result_path, histogram_dir)

        fig_row=0
        for program_chunk in (data_chunks):
            chunk_data = self.datas[program_chunk]
            for processed_unit in (chunk_data.processed_data):
                chunk_info = f'{chunk_data.tpkernel_name} {chunk_data.program_chunk.split("(")[0]} {chunk_data.run_mode} {processed_unit} with np={chunk_data.np} N={chunk_data.N} Nr={chunk_data.Nr} ntest={chunk_data.ntest}'
                heatmap_filename = (f'Heatmap of {chunk_info}'.replace(' ', '-'))
                heatmap_filename = os.path.join(heatmap_dir, heatmap_filename + '.png')
                histogram_filename_all = (f'Histogram of {chunk_info}'.replace(' ', '-'))
                histogram_filename_all = os.path.join(histogram_dir, histogram_filename_all + '.png')
                if os.path.exists(heatmap_filename) and os.path.exists(histogram_filename_all):
                    save_dir = heatmap_dir.replace('result-heatmap', 'result-all')
                    file_name = self.task_name + f'-{chunk_data.program_chunk.split("(")[0]}'+ f'-{processed_unit}'
                    TPcp.merge_two_figures(heatmap_filename=heatmap_filename, histogram_filename=histogram_filename_all, save_dir = save_dir, file_name = file_name)

# A collection of Task_Datas's statistics datas
class Kernels:
    def __init__(self, log_data_all:pd.DataFrame, reslut_dir:str):
        self.log_data_all = log_data_all
        self.kernel_name_list = log_data_all['kernel_name'].unique()
        self.mpi_func_degradation: pd.DataFrame = None
        self.kernel_degradation: pd.DataFrame = None
        self.reslut_dir = reslut_dir
        self.get_mpi_func_degradation()
        self.get_kernel_degradation()

    def get_mpi_func_degradation(self):
        header_mpi_func_degradation = ['kernel_name', 'mpi_func_alg', 'np', 'tpkernel_name', 'compute_unit', 'socket_num', \
                   'data_size_comm(KiB)', 'data_size_GEMM(KiB)', 'data_size_memory(KiB)', 'ntest', 'N', 'Nr', 'processed_unit',\
                    'comm-mean-deg', 'comm-max-deg', 'comm-std_err-deg', 'gemm-mean-deg', 'gemm-max-deg', 'gemm-std_err-deg',\
                    'commonly-total_time_per_step(us)', 'compcomm-total_time_per_step(us)', 'componly_GEMM-total_time_per_step(us)',\
                    'total_time_per_step(us)-theoretical','total_time_per_step(us)-deg' 
                    #{compcomm - componly_GEMM - commonly_comm}/(componly_GEMM + commonly_comm)
                    ]
        mpi_func_degradation_rows:list[list] = []
        for kernel_name in self.kernel_name_list:
            mpi_func_alg_list = self.log_data_all[self.log_data_all['kernel_name'] == kernel_name]['mpi_func_alg'].unique()
            for mpi_func_alg in mpi_func_alg_list:
                data = self.log_data_all[
                    (self.log_data_all['kernel_name'] == kernel_name)
                    & (self.log_data_all['mpi_func_alg'] == mpi_func_alg)
                ]
                if data.empty:
                    print(f"Data not found for {kernel_name} {mpi_func_alg}.")
                    continue
                processed_unit = 'us'
                compute_unit = data['compute_unit'].iloc[0]
                Np = data['np'].iloc[0]
                tpkernel_name = data['tpkernel_name'].iloc[0]
                socket_num = data['socket_num'].iloc[0]

                # Don't support different processed units and compute units for now
                data_f = data[(data['processed_unit'] == processed_unit) & (data['compute_unit'] == compute_unit)]
                if data_f.empty:
                    print(f"data_f not found for {kernel_name} {mpi_func_alg}.")
                    continue

                data_commonly_comm = data_f[(data_f['program_pattern'] == 'commonly') & (data_f['program_chunk'].str.contains('comm')) & (~data_f['program_chunk'].str.contains('GEMM'))]
                data_compcomm_comm = data_f[(data_f['program_pattern'] == 'compcomm') & (data_f['program_chunk'].str.contains('comm')) & (~data_f['program_chunk'].str.contains('GEMM'))]
                data_compcomm_GEMM = data_f[(data_f['program_pattern'] == 'compcomm') & (data_f['program_chunk'].str.contains('GEMM')) & (~data_f['program_chunk'].str.contains('comm'))]
                data_componly_GEMM = data_f[(data_f['program_pattern'] == 'componly') & (data_f['program_chunk'].str.contains('GEMM')) & (~data_f['program_chunk'].str.contains('comm'))]

                if len(data_commonly_comm) != 1 or len(data_compcomm_comm) != 1 or len(data_compcomm_GEMM) != 1 or len(data_componly_GEMM) != 1:
                    print(f"Data program_pattern len error for {kernel_name} {mpi_func_alg}.")
                    print(f"data_commonly_comm: {len(data_commonly_comm)}")
                    print(f"data_compcomm_comm: {len(data_compcomm_comm)}")
                    print(f"data_compcomm_GEMM: {len(data_compcomm_GEMM)}")
                    print(f"data_componly_GEMM: {len(data_componly_GEMM)}")
                    continue

                data_size_comm = data_commonly_comm['data_size_operate(KiB)'].iloc[0]
                data_size_GEMM = data_componly_GEMM['data_size_operate(KiB)'].iloc[0]
                data_size_memory = data_compcomm_GEMM['data_size_memory(KiB)'].iloc[0]
                ntest = data_f['ntest'].iloc[0]
                N = data_f['N'].iloc[0]
                Nr = data_f['Nr'].iloc[0]
                comm_mean_deg:float = (data_compcomm_comm['mean'].iloc[0] - data_commonly_comm['mean'].iloc[0]) / data_commonly_comm['mean'].iloc[0]
                comm_max_deg:float = (data_compcomm_comm['max'].iloc[0] - data_commonly_comm['max'].iloc[0]) / data_commonly_comm['max'].iloc[0]
                comm_std_err_deg:float = (data_compcomm_comm['standard_error'].iloc[0] - data_commonly_comm['standard_error'].iloc[0]) / data_commonly_comm['standard_error'].iloc[0]

                gemm_mean_deg:float = (data_compcomm_GEMM['mean'].iloc[0] - data_componly_GEMM['mean'].iloc[0]) / data_componly_GEMM['mean'].iloc[0]
                gemm_max_deg:float = (data_compcomm_GEMM['max'].iloc[0] - data_componly_GEMM['max'].iloc[0]) / data_componly_GEMM['max'].iloc[0]
                gemm_std_err_deg:float = (data_compcomm_GEMM['standard_error'].iloc[0] - data_componly_GEMM['standard_error'].iloc[0]) / data_componly_GEMM['standard_error'].iloc[0]

                commonly_total_time_per_step:float = data_commonly_comm['total_time_per_step(us)'].iloc[0]
                compcomm_total_time_per_step:float = data_compcomm_comm['total_time_per_step(us)'].iloc[0]
                componly_GEMM_total_time_per_step:float = data_componly_GEMM['total_time_per_step(us)'].iloc[0]

                total_time_per_step_theoretical = (componly_GEMM_total_time_per_step + commonly_total_time_per_step)
                total_time_per_step_deg = (compcomm_total_time_per_step - total_time_per_step_theoretical) / total_time_per_step_theoretical

                mpi_func_degradation_rows.append([kernel_name, mpi_func_alg, Np, tpkernel_name, compute_unit, socket_num,
                     data_size_comm, data_size_GEMM, data_size_memory, ntest, N, Nr, processed_unit, 
                     comm_mean_deg, comm_max_deg, comm_std_err_deg, gemm_mean_deg, gemm_max_deg, gemm_std_err_deg, 
                     commonly_total_time_per_step, compcomm_total_time_per_step, componly_GEMM_total_time_per_step, 
                     total_time_per_step_theoretical ,total_time_per_step_deg])

        self.mpi_func_degradation = pd.DataFrame(mpi_func_degradation_rows, columns=header_mpi_func_degradation)
        self.mpi_func_degradation = self.mpi_func_degradation.sort_values(by=['tpkernel_name', 'np', 'mpi_func_alg', 'N', 'Nr'], ignore_index=True)
        self.mpi_func_degradation.to_csv(os.path.join(self.reslut_dir, f'{Chunk_Data.machine}-mpi_func-degradation.csv'), index=False)

    def get_kernel_degradation(self):
        header_kernel_degradation =  ['kernel_name', 'np', 'tpkernel_name', 'compute_unit', 'socket_num', 
            'data_size_comm(KiB)', 'data_size_GEMM(KiB)', 'data_size_memory(KiB)', 'ntest', 'N', 'Nr', 'processed_unit',
            'commonly-best_func_alg', 'commonly-comm-mean-adv', 'commonly-comm-std_err-adv', 'commonly-comm-max-adv', # best func alg is the algorithm whose total_time_per_step is the smallest
            'compcomm-best_func_alg', 'compcomm-comm-mean-adv', 'compcomm-comm-std_err-adv', 'compcomm-comm-max-adv', 
            'compcomm-GEMM-mean-adv', 'compcomm-GEMM-std_err-adv', 'compcomm-GEMM-max-adv',
            'compcomm-GEMMcomm-mean-adv', 'compcomm-GEMMcommstd_err-adv', 'compcomm-GEMMcomm-max-adv',
            'total_time_per_step(us)-theoretical-adv', 'commonly-total_time_per_step(us)-adv', 'compcomm-total_time_per_step(us)-adv' 
            #{compcomm(best)-componly_GEMM-commonly_comm(best)}/(componly_GEMM + commonly_comm(best))
        ]

        kernel_degradation_rows:list[list] = []
        if self.mpi_func_degradation is None:
            self.get_mpi_func_degradation()
        if self.mpi_func_degradation is None:
            exit("MPI function degradation not found.")
        for kernel_name in self.kernel_name_list:
            data = self.mpi_func_degradation[self.mpi_func_degradation['kernel_name'] == kernel_name]
            if data.empty:
                print(f"Data not found for {kernel_name}.")
                continue
            Np = data['np'].iloc[0]
            tpkernel_name = data['tpkernel_name'].iloc[0]
            compute_unit = data['compute_unit'].iloc[0]
            socket_num = data['socket_num'].iloc[0]
            data_size_comm = data['data_size_comm(KiB)'].iloc[0]
            data_size_GEMM = data['data_size_GEMM(KiB)'].iloc[0]
            data_size_memory = data['data_size_memory(KiB)'].iloc[0]
            ntest = data['ntest'].iloc[0]
            N = data['N'].iloc[0]
            Nr = data['Nr'].iloc[0]
            processed_unit = data['processed_unit'].iloc[0]
            # Get the smallest commonly-total_time_per_step(us)'s 'mpi_func_alg' in all data
            commonly_best_row = data.loc[data['commonly-total_time_per_step(us)'].idxmin()]
            compcomm_best_row = data.loc[data['compcomm-total_time_per_step(us)'].idxmin()]
            commonly_best_func_alg = commonly_best_row['mpi_func_alg']
            compcomm_best_func_alg = compcomm_best_row['mpi_func_alg']

            # 
            log_data = self.log_data_all[
                (self.log_data_all["kernel_name"] == kernel_name) &
                (self.log_data_all["processed_unit"] == processed_unit) &
                (self.log_data_all["compute_unit"] == compute_unit) &
                (self.log_data_all["socket_num"] == socket_num) &
                (self.log_data_all["program_chunk"].str.contains('comm')) &
                (~self.log_data_all["program_chunk"].str.contains('GEMM'))
            ]

            commonly_best_func_alg_commonly = log_data[
                (log_data["mpi_func_alg"] == commonly_best_func_alg) &
                (log_data["program_pattern"] == "commonly")
            ]
            commonly_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == commonly_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]

            compcomm_best_func_alg_commonly = log_data[
                (log_data["mpi_func_alg"] == compcomm_best_func_alg) &
                (log_data["program_pattern"] == "commonly")
            ]
            compcomm_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == compcomm_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]


            if len(commonly_best_func_alg_compcomm) != 1 or len(commonly_best_func_alg_commonly) != 1 or len(compcomm_best_func_alg_commonly) != 1 or len(compcomm_best_func_alg_compcomm) != 1:
                print(f"Data program_pattern len error for {kernel_name} {commonly_best_func_alg}.")
                print(f"commonly_best_func_alg_compcomm: {len(commonly_best_func_alg_compcomm)}")
                print(f"commonly_best_func_alg_commonly: {len(commonly_best_func_alg_commonly)}")
                print(f"compcomm_best_func_alg_commonly: {len(compcomm_best_func_alg_commonly)}")
                print(f"compcomm_best_func_alg_compcomm: {len(compcomm_best_func_alg_compcomm)}")
                continue

            # How fast the comm is advanced in commonly between commonly_best_func_alg and compcomm_best_func_alg
            commonly_comm_mean_adv = (compcomm_best_func_alg_commonly['mean'].iloc[0] - commonly_best_func_alg_commonly['mean'].iloc[0]) / compcomm_best_func_alg_commonly['mean'].iloc[0]
            commonly_comm_std_err_adv = (compcomm_best_func_alg_commonly['standard_error'].iloc[0] - commonly_best_func_alg_commonly['standard_error'].iloc[0]) / compcomm_best_func_alg_commonly['standard_error'].iloc[0]
            commonly_comm_max_adv = (compcomm_best_func_alg_commonly['max'].iloc[0] - commonly_best_func_alg_commonly['max'].iloc[0]) / compcomm_best_func_alg_commonly['max'].iloc[0]


            # How fast the comm is advanced in compcomm between commonly_best_func_alg and compcomm_best_func_alg
            compcomm_comm_mean_adv = (commonly_best_func_alg_compcomm['mean'].iloc[0] - compcomm_best_func_alg_compcomm['mean'].iloc[0]) / commonly_best_func_alg_compcomm['mean'].iloc[0]
            compcomm_comm_std_err_adv = (commonly_best_func_alg_compcomm['standard_error'].iloc[0] - compcomm_best_func_alg_compcomm['standard_error'].iloc[0]) / commonly_best_func_alg_compcomm['standard_error'].iloc[0]
            compcomm_comm_max_adv = (commonly_best_func_alg_compcomm['max'].iloc[0] - compcomm_best_func_alg_compcomm['max'].iloc[0]) / commonly_best_func_alg_compcomm['max'].iloc[0]


            log_data = self.log_data_all[
                (self.log_data_all["kernel_name"] == kernel_name) &
                (self.log_data_all["processed_unit"] == processed_unit) &
                (self.log_data_all["compute_unit"] == compute_unit) &
                (self.log_data_all["socket_num"] == socket_num) &
                (~self.log_data_all["program_chunk"].str.contains('comm')) &
                (self.log_data_all["program_chunk"].str.contains('GEMM'))
            ]

            commonly_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == commonly_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]
            compcomm_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == compcomm_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]

            # How fast the GEMM is advanced in compcomm between commonly_best_func_alg and compcomm_best_func_alg
            compcomm_GEMM_mean_adv = (commonly_best_func_alg_compcomm['mean'].iloc[0] - compcomm_best_func_alg_compcomm['mean'].iloc[0]) / commonly_best_func_alg_compcomm['mean'].iloc[0]
            compcomm_GEMM_std_err_adv = (commonly_best_func_alg_compcomm['standard_error'].iloc[0] - compcomm_best_func_alg_compcomm['standard_error'].iloc[0]) / commonly_best_func_alg_compcomm['standard_error'].iloc[0]
            compcomm_GEMM_max_adv = (commonly_best_func_alg_compcomm['max'].iloc[0] - compcomm_best_func_alg_compcomm['max'].iloc[0]) / commonly_best_func_alg_compcomm['max'].iloc[0]

            log_data = self.log_data_all[
                (self.log_data_all["kernel_name"] == kernel_name) &
                (self.log_data_all["processed_unit"] == processed_unit) &
                (self.log_data_all["compute_unit"] == compute_unit) &
                (self.log_data_all["socket_num"] == socket_num) &
                (self.log_data_all["program_chunk"].str.contains('GEMMcomm'))
            ]

            commonly_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == commonly_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]
            compcomm_best_func_alg_compcomm = log_data[
                (log_data["mpi_func_alg"] == compcomm_best_func_alg) &
                (log_data["program_pattern"] == "compcomm")
            ]

            # How fast the GEMMcomm is advanced in compcomm between commonly_best_func_alg and compcomm_best_func_alg
            compcomm_GEMMcomm_mean_adv = (commonly_best_func_alg_compcomm['mean'].iloc[0] - compcomm_best_func_alg_compcomm['mean'].iloc[0]) / commonly_best_func_alg_compcomm['mean'].iloc[0]
            compcomm_GEMMcomm_std_err_adv = (commonly_best_func_alg_compcomm['standard_error'].iloc[0] - compcomm_best_func_alg_compcomm['standard_error'].iloc[0]) / commonly_best_func_alg_compcomm['standard_error'].iloc[0]
            compcomm_GEMMcomm_max_adv = (commonly_best_func_alg_compcomm['max'].iloc[0] - compcomm_best_func_alg_compcomm['max'].iloc[0]) / commonly_best_func_alg_compcomm['max'].iloc[0]
            
            total_time_per_step_theoretical_adv = (compcomm_best_row['compcomm-total_time_per_step(us)'] 
                        - commonly_best_row['total_time_per_step(us)-theoretical']
                        ) / compcomm_best_row['compcomm-total_time_per_step(us)'] 

            commonly_total_time_per_step_adv = (compcomm_best_func_alg_commonly['total_time_per_step(us)'].iloc[0] - commonly_best_func_alg_commonly['total_time_per_step(us)'].iloc[0]) / compcomm_best_func_alg_commonly['total_time_per_step(us)'].iloc[0]
            compcomm_total_time_per_step_adv = (commonly_best_func_alg_compcomm['total_time_per_step(us)'].iloc[0] - compcomm_best_func_alg_compcomm['total_time_per_step(us)'].iloc[0]) / commonly_best_func_alg_compcomm['total_time_per_step(us)'].iloc[0]
            
            kernel_degradation_rows.append([kernel_name, Np, tpkernel_name, compute_unit, socket_num,
                data_size_comm, data_size_GEMM, data_size_memory, ntest, N, Nr, processed_unit,
                commonly_best_func_alg, commonly_comm_mean_adv, commonly_comm_std_err_adv, commonly_comm_max_adv,
                compcomm_best_func_alg, compcomm_comm_mean_adv, compcomm_comm_std_err_adv, compcomm_comm_max_adv,
                compcomm_GEMM_mean_adv, compcomm_GEMM_std_err_adv, compcomm_GEMM_max_adv,
                compcomm_GEMMcomm_mean_adv, compcomm_GEMMcomm_std_err_adv, compcomm_GEMMcomm_max_adv,
                total_time_per_step_theoretical_adv, commonly_total_time_per_step_adv, compcomm_total_time_per_step_adv
            ])

        self.kernel_degradation = pd.DataFrame(kernel_degradation_rows, columns=header_kernel_degradation)
        self.kernel_degradation = self.kernel_degradation.sort_values(by=['tpkernel_name', 'np', 'N', 'Nr'], ignore_index=True)
        self.kernel_degradation.to_csv(os.path.join(self.reslut_dir, f'{Chunk_Data.machine}-kernel-degradation.csv'), index=False)


    def draw_distribution_best_mpi_func_alg(self, task_names=[], program_chunks=[], processed_units_dict={}, save_dir=''):
        '''
        np = 64, 128
        二维图x2：（分别是在commonly和compcomm下的算法分布）
            - data: self.kernel_degradation
            - 横坐标：N
            - 纵坐标：Nr
            - z：颜色表示算法，数字表示提升
        '''
        data_to_plot = self.kernel_degradation[self.kernel_degradation['Nr'] != 512]
        for tpkernel_name in self.kernel_degradation['tpkernel_name'].unique():
            data_to_plot = self.kernel_degradation[self.kernel_degradation['tpkernel_name'] == tpkernel_name]
            if data_to_plot.empty:
                continue
            for Np in data_to_plot['np'].unique():
                data_to_plot_np = data_to_plot[data_to_plot['np'] == Np]
                if data_to_plot_np.empty:
                    continue
                title_name = f'{tpkernel_name}\n Best MPI Function Algorithm Distribution with np={Np}'
                file_name_base = f'{tpkernel_name} Best MPI Function Algorithm Distribution np={Np}'.replace(' ', '-')
                directory = os.path.join(save_dir, f'{Chunk_Data.machine}-{Chunk_Data.time_stamp}-result-mpi_func_alg_best-distribution')
                for z_value_col in ['total_time_per_step(us)-adv', 'comm-max-adv', 'comm-mean-adv']:
                    file_name = file_name_base + f'-{z_value_col}'
                    TPcp.draw_distribution_best_mpi_func_alg(data_to_plot = data_to_plot_np, title_name = title_name, z_value_col = z_value_col, save_dir = directory, file_name = file_name, mode='comm')
                for mode in ['GEMM', 'GEMMcomm']:
                    file_name = file_name_base + f'-{mode}-adv'
                    TPcp.draw_distribution_best_mpi_func_alg(data_to_plot = data_to_plot_np, title_name = title_name, z_value_col = z_value_col, save_dir = directory, file_name = file_name, mode=mode)
        pass

    def draw_statistic_scalability(self, task_names=[], program_chunks=[], processed_units_dict={}, save_dir=''):
        '''
        Data: self.mpi_func_degradation
        TimeToSolution的拓展性（Nr=10）
            - commonly
            - compcomm
            - componly
        '''
        for Nr in self.mpi_func_degradation['Nr'].unique():
            if Nr == 512:
                continue
            data_to_plot = self.mpi_func_degradation[self.mpi_func_degradation['Nr'] == Nr]
            for tpkernel_name in data_to_plot['tpkernel_name'].unique():
                data_to_plot_kernel = data_to_plot[data_to_plot['tpkernel_name'] == tpkernel_name]
                if data_to_plot_kernel.empty:
                    print(f"Data not found for {tpkernel_name}.")
                    continue
                for Np in data_to_plot_kernel['np'].unique():
                    data_to_plot_np = data_to_plot_kernel[data_to_plot_kernel['np'] == Np]
                    if data_to_plot_np.empty:
                        print(f"Data not found for {tpkernel_name} np={Np}.")
                        continue
                    ntest=data_to_plot_np['ntest'].iloc[0]
                    for mpi_func_alg in data_to_plot_np['mpi_func_alg'].unique():
                        data_to_plot_alg = data_to_plot_np[data_to_plot_np['mpi_func_alg'] == mpi_func_alg]
                        if data_to_plot_alg.empty:
                            print(f"Data not found for {tpkernel_name} np={Np} alg={mpi_func_alg}.")
                            continue
                        title_name = f'{tpkernel_name}\nTime/Step scalability {mpi_func_alg} Np={Np} ntest={ntest} Nr={Nr}'
                        file_name_base = f'{tpkernel_name} time_per_step scalability {mpi_func_alg} Np={Np} ntest={ntest} Nr={Nr}'.replace(' ', '-')
                        directory = os.path.join(save_dir, f'{Chunk_Data.machine}-{Chunk_Data.time_stamp}-result-scalability')
                        TPcp.draw_statistic_scalability(data_to_plot = data_to_plot_alg, title_name = title_name, save_dir = directory, file_name = file_name_base)

    def draw_statistic_figures(self, task_names=[], program_chunks=[], processed_units_dict={}, save_dir=''):
        '''
        统计结果分析内容：
        - 固定算法，去探究该算法comm的数据规模的可拓展性
            - 带数据点的折线图
            - 纵坐标：
                - 图1-1: us的mean、max、min、standard_error
                - 图1-2: KB_per_sec的mean、max、min、standard_error
            - 横坐标：
                - data_size_operate(KiB)
        - 固定算法，探究不同N之间，compcomm对比commonly，comm的劣化情况
            - 带数据点的折线图
            - 纵坐标：
                - 图1-1: comm的us的mean、standard_error
                - 图1-2: comm的us的max、min
                - 图2-1: KB_per_sec的mean、standard_error
                - 图2-2: KB_per_sec的max、min
            - 横坐标
                - data_size_operate(KiB)
        - 固定N，探究不同算法之间，comm和gemm的差异
            - 带数据点的折线图
            - 纵坐标：
                - 图1-1: comm的us的mean、max、min、standard_error
                - 图1-2: GEMM的us的mean、max、min、standard_error
                - 图2-1: comm的KB_per_sec的mean、max、min、standard_error
                - 图2-2: GEMM的KB_per_sec的mean、max、min、standard_error
            - 横坐标：
                - 不同的MPI通讯算法
        - 固定N，探究不同算法之间，compcomm对比commonly，comm的劣化情况
            - 带数据点的折线图
            - 纵坐标：
                - 图1-1: comm的us的mean、standard_error
                - 图1-2: comm的us的max、min
                - 图2-1: KB_per_sec的mean、standard_error
                - 图2-2: KB_per_sec的max、min
            - 横坐标：
                - 不同的MPI通讯算法
        '''
        save_dir = os.path.join(save_dir, f'{Chunk_Data.machine}-{Chunk_Data.time_stamp}-result-statistic')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # if len(task_names) == 0:
        #     task_names = self.task_datas.keys()
        kernel_list = self.log_data_all['tpkernel_name'].unique()
        for tpkernel_name in kernel_list:
            mpi_func_alg_list = self.log_data_all[self.log_data_all['tpkernel_name'] == tpkernel_name]['mpi_func_alg'].unique()
            for mpi_func_alg in mpi_func_alg_list:
                Nr = 10
                # Explore
                run_mode_list = self.log_data_all[(self.log_data_all['tpkernel_name'] == tpkernel_name) & (self.log_data_all['mpi_func_alg'] == mpi_func_alg)]['run_mode'].unique()
                for run_mode in run_mode_list:
                    program_chunk_list= self.log_data_all[(self.log_data_all['tpkernel_name'] == tpkernel_name) & (self.log_data_all['mpi_func_alg'] == mpi_func_alg) & (self.log_data_all['run_mode'] == run_mode)]['program_chunk'].unique()
                    for program_chunk in program_chunk_list:
                        data_to_plot = self.log_data_all[
                            (self.log_data_all['tpkernel_name'] == tpkernel_name) 
                            & (self.log_data_all['mpi_func_alg'] == mpi_func_alg)
                            & (self.log_data_all['run_mode'] == run_mode)
                            & (self.log_data_all['program_chunk'] == program_chunk)
                            & (self.log_data_all['Nr'] == Nr)
                        ]
                        if data_to_plot.empty:
                            continue
                        ntest = data_to_plot['ntest'].iloc[0]
                        chunk=program_chunk.split('(')[0]
                        title_name = f'{tpkernel_name} {run_mode}\n{chunk} ntest={ntest} Nr={Nr} all_comm_statistic'
                        file_name_base = f'all_comm_statistic {tpkernel_name} {run_mode} {chunk} ntest={ntest} Nr={Nr}'.replace(' ', '-').replace(f'\n', '-')
                        TPcp.draw_statistic_all_comm(data_to_plot = data_to_plot, x="data_size_operate(KiB)", title_name = title_name, file_name_base = file_name_base, save_dir = save_dir)
                
                # Explore 'compcomm' vs 'commonly' in 'comm'
                data_to_plot = self.log_data_all[
                    (self.log_data_all['tpkernel_name'] == tpkernel_name)
                    & (self.log_data_all['mpi_func_alg'] == mpi_func_alg)
                    & (self.log_data_all['program_chunk'].str.contains('comm'))
                    & (~self.log_data_all['program_chunk'].str.contains('GEMM'))
                    & (self.log_data_all['program_pattern'].isin(['compcomm', 'commonly']))
                    & (self.log_data_all['Nr'] == Nr)
                ]
                if data_to_plot.empty:
                    continue
                ntest = data_to_plot['ntest'].iloc[0]
                compute_unit = data_to_plot['compute_unit'].iloc[0]
                title_name = f'{tpkernel_name} {compute_unit} {mpi_func_alg} ntest={ntest} Nr={Nr}\ncompcomm vs. commonly'
                file_name_base = f'compare_CommonlyCompcomm_statistic {tpkernel_name} {compute_unit} {mpi_func_alg} ntest={ntest} Nr={Nr}'.replace(' ', '-').replace(f'\n', '-')
                TPcp.draw_statistic_compare_CommonlyCompcomm(data_to_plot = data_to_plot, x="data_size_operate(KiB)", title_name = title_name, file_name_base = file_name_base, save_dir = save_dir)
            
            N_list = self.log_data_all[self.log_data_all['tpkernel_name'] == tpkernel_name]['N'].unique()
            for N in N_list:
                Nr = 10
                data_to_plot = self.log_data_all[
                    (self.log_data_all['tpkernel_name'] == tpkernel_name) 
                    & (self.log_data_all['N'] == N)
                    & (self.log_data_all['program_pattern'] == 'compcomm')
                    & (self.log_data_all['Nr'] == Nr)
                ]
                if data_to_plot.empty:
                    continue
                compute_unit = data_to_plot['compute_unit'].iloc[0]
                program_pattern = 'compcomm'
                ntest = data_to_plot['ntest'].iloc[0]
                title_name = f'{tpkernel_name} {compute_unit} {program_pattern} N={N}\nntest={ntest} compare_CommComp_statistic'
                file_name_base = f'compare_CommComp_statistic {tpkernel_name} {compute_unit} {program_pattern} ntest={ntest} N={N} Nr={Nr}'.replace(' ', '-').replace(f'\n', '-')
                TPcp.draw_statistic_compare_CommComp(data_to_plot = data_to_plot, x="mpi_func_alg", title_name = title_name, file_name_base = file_name_base, save_dir = save_dir)

                data_to_plot = self.log_data_all[
                    (self.log_data_all['tpkernel_name'] == tpkernel_name)
                    & (self.log_data_all['N'] == N)
                    & (self.log_data_all['program_chunk'].str.contains('comm'))
                    & (~self.log_data_all['program_chunk'].str.contains('GEMM'))
                    & (self.log_data_all['program_pattern'].isin(['compcomm', 'commonly']))
                    & (self.log_data_all['Nr'] == Nr)
                ]
                if data_to_plot.empty:
                    continue
                ntest = data_to_plot['ntest'].iloc[0]
                title_name = f'{tpkernel_name} {compute_unit} ntest={ntest}\nN={N} Nr={Nr}compcomm vs.commonly'
                file_name_base = f'compare_CommonlyCompcomm_statistic {tpkernel_name} {compute_unit} ntest={ntest} N={N} Nr={Nr}'.replace(' ', '-').replace(f'\n', '-')
                TPcp.draw_statistic_compare_CommonlyCompcomm(data_to_plot = data_to_plot, x="mpi_func_alg", title_name = title_name, file_name_base = file_name_base, save_dir = save_dir)
        


class Tasks:
    def __init__(self, file_path:str, generate_summary:bool = False, generate_GEMMcomm:bool = False):
        self.task_datas:dict[str, Task_Datas] = {}
        self.invalid_files = []
        self.file_path = file_path
        self.log_path = os.path.join(file_path, 'log')
        self.log_data_all: pd.DataFrame = None
        self.kernels:Kernels = None
        if generate_GEMMcomm == True:
            self.generate_GEMMcomm()
        self._get_all_tasks(generate_summary)

    def _get_all_tasks(self, generate_summary:bool = False):
        # Populate self.task_set and self.task_datas based on log_path
        # Scan all files in log_path
        summary_name = Chunk_Data.machine + '-result_all' + '.csv'
        summary_path = os.path.join(os.path.dirname(self.log_path), summary_name)

        headers = ['kernel_name', 'np', 'tpkernel_name', 'run_mode', 'compute_unit', 'program_pattern', 'socket_num', 'mpi_func_alg', \
                   'data_size_operate(KiB)', 'data_size_memory(KiB)', 'ntest', 'N', 'Nr','program_chunk', 'processed_unit', \
                    'mean', 'min', 'max', '50%', 'standard_error', 'total_time_per_step(us)']
        summary_rows = []

        for file_name in os.listdir(self.log_path):
            if file_name.endswith('.csv'):
                raw_data = pd.read_csv(os.path.join(self.log_path, file_name), header=0, index_col=0, dtype=str)
                Np = int(re.search(r'np(\d+)', file_name).group(1))
                ntest = int(re.search(r'ntest(\d+)', file_name).group(1))
                if raw_data.shape != (Np, ntest):
                    print(f"Invalid data shape for {file_name}. Expected ({Np}, {ntest}), got {raw_data.shape}.")
                    self.invalid_files.append(os.path.join(self.log_path, file_name))
                    continue
                raw_data = raw_data[raw_data.index.to_series().apply(lambda x: re.match(r'rank\d+', x) is not None)]
                raw_data = raw_data.astype(int)
                task_name, program_chunk, raw_unit = self._extract_file_name(file_name)
                self._add_kernel_data(task_name, program_chunk, raw_unit, raw_data)

                run_mode = task_name.split('-')[2]
                kernel_name = task_name.replace(f"-{run_mode}", '')

                if (generate_summary == True):
                    chunk_data = self.task_datas[task_name].datas[program_chunk]
                    for processed_unit in chunk_data.processed_data:
                        NP = chunk_data.np
                        tpkernel_name = chunk_data.tpkernel_name
                        run_mode = chunk_data.run_mode
                        compute_unit, program_pattern, socket, mpi_alg = chunk_data.run_mode_sub.get_run_mode_sub()
                        ntest = chunk_data.ntest
                        N = chunk_data.N
                        Nr = chunk_data.Nr
                        program_chunk = chunk_data.program_chunk
                        data_size_operate = chunk_data.data_size_operate
                        data_size_memory = chunk_data.data_size_memory
                        data = pd.DataFrame(chunk_data.processed_data[processed_unit].values.flatten(), columns=['values'])
                        standard_error = np.std(data['values'], ddof=1) / np.sqrt(len(data['values']))
                        mean_val = np.mean(chunk_data.processed_data[processed_unit].values.flatten())
                        min_val = np.min(chunk_data.processed_data[processed_unit].values.flatten())
                        max_val = np.max(chunk_data.processed_data[processed_unit].values.flatten())
                        mid_val = np.median(chunk_data.processed_data[processed_unit].values.flatten())
                        total_time_per_step = chunk_data.total_time_per_step
                        summary_rows.append([kernel_name, NP, tpkernel_name, run_mode, compute_unit, program_pattern, socket, mpi_alg,\
                                            data_size_operate, data_size_memory, ntest, N, Nr, program_chunk, processed_unit,\
                                                mean_val, min_val, max_val, mid_val, standard_error, total_time_per_step])
        if (generate_summary == True):
            self.log_data_all = pd.DataFrame(summary_rows, columns=headers)
            self.log_data_all = self.log_data_all.sort_values(by=['np', 'mpi_func_alg', 'run_mode', 'program_chunk', 'processed_unit', 'N', 'Nr'])
            self.log_data_all.to_csv(summary_path, index=False)
            self.kernels = Kernels(log_data_all = self.log_data_all, reslut_dir = os.path.dirname(self.log_path))

    def generate_GEMMcomm(self):
        '''
        Generate a new program chunk 'GEMMcomm' based on 'GEMM' and 'comm' program chunks of compcomm kernels
        '''
        task_sets=set()
        program_chunks = ['GEMM(ns)', 'comm(ns)']
        for file_name in os.listdir(self.log_path):
            if file_name.endswith('.csv') and 'compcomm' in file_name:
                task_name, program_chunk, raw_unit = self._extract_file_name(file_name)
                if task_name not in task_sets:
                    task_sets.add(task_name)
                    GEMM_data:pd.DataFrame = None
                    comm_data:pd.DataFrame = None
                    for program_chunk in program_chunks:
                        file_name_generate = self.get_file_name(task_name = task_name, program_chunk = program_chunk)
                        raw_data = pd.read_csv(os.path.join(self.log_path, file_name_generate), header=0, index_col=0, dtype=str)
                        Np = int(re.search(r'np(\d+)', file_name).group(1))
                        ntest = int(re.search(r'ntest(\d+)', file_name).group(1))
                        if raw_data.shape != (Np, ntest):
                            print(f"Invalid data shape for {file_name}. Expected ({Np}, {ntest}), got {raw_data.shape}.")
                            self.invalid_files.append(os.path.join(self.log_path, file_name))
                            continue
                        raw_data = raw_data[raw_data.index.to_series().apply(lambda x: re.match(r'rank\d+', x) is not None)]
                        raw_data = raw_data.astype(int)
                        if 'GEMM' in program_chunk:
                            GEMM_data = raw_data
                        elif 'comm' in program_chunk:
                            comm_data = raw_data
                    if GEMM_data is not None and comm_data is not None:
                        GEMMcomm_data = GEMM_data + comm_data
                        file_name_generate = self.get_file_name(task_name = task_name, program_chunk = 'GEMMcomm(ns)')
                        GEMMcomm_data.to_csv(os.path.join(self.log_path, file_name_generate), index=True)
        pass

    def _extract_file_name(self, file_name):
        # Extract task set and program chunk from file_name
        # file_name = np{np}-{tpkernel_name}-{run_mode}-{program_chunk}({counter_unit})-ntest{ntest}-N{N}-Nr{Nr}.csv
        # task_name = np{np}-{tpkernel_name}-{run_mode}-ntest{ntest}-N{N}-Nr{Nr}
        program_chunk = file_name.split('-')[3]
        raw_unit = file_name.split('(')[1].split(')')[0]
        task_name = file_name.replace(f'-{program_chunk}', '').replace('.csv', '')
        return task_name, program_chunk, raw_unit

    def delete_invalid_files(self):
        tasks = set()
        res_dir=''
        for file_name in self.invalid_files:
            program_chunk = file_name.split('-')[3]
            task_name = file_name.replace(f'-{program_chunk}', '').replace('.csv', '')
            tasks.add(task_name)
            res_dir = os.path.dirname(file_name) if res_dir == '' else res_dir
        for file in os.listdir(res_dir):
            program_chunk = file.split('-')[3]
            task_name = file.replace(f'-{program_chunk}', '').replace('.csv', '')
            if task_name in tasks:
                os.remove(os.path.join(res_dir, file))
                print(f"Deleted {file}")
        
    def get_all_task_names(self):
        return self.task_datas.keys()

    def get_task_name(self, tpkernel_name:str, np:int, run_mode:str, ntest:int, N:int, Nr:int):
        name = f"np{np}-{tpkernel_name}-{run_mode}-ntest{ntest}-N{N}-Nr{Nr}"
        if name not in self.task_datas:
            return None
        return name

    def get_file_name(self, task_name:str = None, kernel_name:str = None, run_mode:str = None, program_chunk:str = None) -> str:
        file_name:str = None
        if program_chunk is None:
            exit("Please input program_chunk such as 'comm(ns)', 'GEMM(cy)' etc.")
        if task_name is not None:
            task_name_split = task_name.split('-')
            task_name_split.insert(3, program_chunk)
            file_name = '-'.join(task_name_split) + '.csv'
        elif kernel_name is not None and run_mode is not None:
            kernel_name_split = kernel_name.split('-')
            kernel_name_split.insert(2, run_mode)
            kernel_name_split.insert(3, program_chunk)
        return file_name
    

    def get_task_names(self, tpkernel_names:list[str], nps:list[int], run_modes:list[str], ntests:list[int], Ns:list[int], Nrs:list[int]):
        task_names = []
        for tpkernel_name in tpkernel_names:
            for np in nps:
                for run_mode in run_modes:
                    for ntest in ntests:
                        for N in Ns:
                            for Nr in Nrs:
                                name = self.get_task_name(tpkernel_name, np, run_mode, ntest, N, Nr)
                                if name is not None:
                                    task_names.append(name)
        return task_names


    def _add_kernel_data(self, task_name, program_chunk, raw_unit, raw_data):
        if task_name not in self.task_datas:
            self.task_datas[task_name] = Task_Datas(self.file_path, task_name, program_chunk, raw_unit, raw_data)
        else:
            self.task_datas[task_name].add_data(program_chunk, raw_unit, raw_data)

    def draw_figures(self, task_names=[], program_chunks=[], processed_units_dict={}, save_dir='', save_mode='old', figure_mode='fixed'):
        if save_mode == 'new':
            Chunk_Data.update_time_stamp()
        elif save_mode == 'old':
            Chunk_Data.get_latest_time_stamp(save_dir)
        else:
            exit(f"Invalid save mode = {save_mode}.")

        if len(task_names) == 0:
            task_names = self.task_datas.keys()
        if figure_mode in ['fixed', 'interactive']:
            for task_name in task_names:
                if task_name in self.task_datas:
                    self.task_datas[task_name].draw_figures(program_chunks = program_chunks, processed_units_dict = processed_units_dict, figure_mode=figure_mode, save_dir = save_dir)
        elif figure_mode in ['statistic']:
            if self.kernels is not None:
                self.kernels.draw_statistic_figures(task_names = task_names, program_chunks = program_chunks, processed_units_dict = processed_units_dict, save_dir = save_dir)
            else:
                exit("Summary Kernels not found.")
        elif figure_mode == 'distribution_best_mpi_func_alg':
            if self.kernels is not None:
                self.kernels.draw_distribution_best_mpi_func_alg(task_names = task_names, program_chunks = program_chunks, processed_units_dict = processed_units_dict, save_dir = save_dir)
            else:
                exit("Summary Kernels not found.")
        elif figure_mode == 'statistic_scalability':
            if self.kernels is not None:
                self.kernels.draw_statistic_scalability(task_names = task_names, program_chunks = program_chunks, processed_units_dict = processed_units_dict, save_dir = save_dir)
            else:
                exit("Summary Kernels not found.")
        else:
            exit(f"Invalid figure mode = {figure_mode}. Please input 'fixed', 'interactive' or 'statistic'.")
            
    def draw_all_figures(self, task_names:list = [], program_chunks:list = [], processed_units_dict: dict = {str:str}, save_dir:str = ''):
        for task_name in task_names:
            self.task_datas[task_name].draw_all_chunks(program_chunks = program_chunks, processed_units_dict = processed_units_dict, result_path = save_dir)

    def generate_summary(self, result_path):
        # Generate a summary file in a markdown format
        # Summary file contains the following information:
        # Insert all the figures' paths in '{machine}-{time_stamp}-result-all' directory
        def sort_key(s):
            parts = s.split('.')[0].split('-')
            key = []
            for part in parts:
                if part.startswith('np'):
                    num = int(part[2:])
                    key.append(('np', num))
                elif part.startswith('ntest'):
                    num = int(part[5:])
                    key.append(('ntest', num))
                elif re.match(r'N\d+', part):
                    num = int(part[1:])
                    key.append(('N', num))
                elif part in ['GEMM', 'Jacobi', 'comm']:
                    key.append(('chunk',part))
                elif part in ['us', 'KB_per_sec', 'GFLOPs', 'bytes_per_cycle']:
                    key.append(('unit', part))
                else:
                    key.append((part,))
            priority_order = {'chunk': 0, 'unit': 1, 'np': 2, 'ntest': 3, 'N': 4}
            key.sort(key=lambda x: priority_order.get(x[0], 4))

            return key

        machine = Chunk_Data.machine
        local_dirs = []
        for d in os.listdir(result_path):
            if os.path.isdir(os.path.join(result_path, d)):
                if d.startswith(f'{machine}-') and d.endswith('result-all'):
                    local_dirs.append(d)

        figures_dir = sorted(local_dirs, key=lambda x: x.split('-')[1], reverse=True)[0]
        summary_name = figures_dir + '.md'
        summary_path = os.path.join(result_path, summary_name)
        figures_path = os.path.join(result_path, figures_dir)

        file_list = os.listdir(figures_path)
        file_list = [f for f in file_list if f.endswith('.png')]
        file_list = sorted(file_list, key = sort_key)

        with open(summary_path, 'w') as f:
            for file_name in file_list:
                f.write(f'![{file_name}]({os.path.join(figures_dir, file_name)})\n') 
