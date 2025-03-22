import os
import sys
import numpy as np
import pandas as pd
import xlsxwriter
import datetime


def add_header(csv_name):
    gemm_end = '数据单位：通信时间中位数(us), 通信的行数：Nr=10N表示方阵的长宽, commcomm和commonly分别表示交错计算和通信和仅进行通信'
    if csv_name == 'gemm_allreduce' :
        return 'GEMM+Allreduce, ' + gemm_end
    elif csv_name == 'gemm_bcast':
        return 'GEMM+bcast, ' + gemm_end
    elif csv_name == 'jacobi2d5p_sendrecv':
        return 'Jacobi2d5p+Sendrecv, 数据单位：通信时间中位数(us), N表示方阵的长宽, rank i与i-1和i+1交换头尾两行, commcomm和commonly分别表示交错计算和通信和仅进行通信;'

def csv_to_excel(writer, output_dir, csv_dir, headers, mode='split'):
    os.makedirs(output_dir, exist_ok=True)
    result_dir = os.path.join(output_dir, csv_dir)
    workbook = writer.book
    font_format = workbook.add_format({'font_name': 'calibri', 'text_wrap': True})
    get_file_name = lambda file_name: os.path.splitext(file_name)[0]

    if mode == 'split':
        for csv_file in os.listdir(result_dir):
            if csv_file.endswith('.csv'):
                csv_name = get_file_name(csv_file)
                csv_path = os.path.join(result_dir, csv_file)
                df = pd.read_csv(csv_path, header=None)
                if headers is not None:
                    df.columns = headers
                else:
                    df.columns = df.iloc[0]
                    df = df[1:]
                    df.insert(0, add_header(csv_name), [f'rank{i}' for i in range(len(df))])
                sheet_name = os.path.splitext(csv_file)[0]
                df.to_excel(writer, sheet_name=csv_name, index=False)
                worksheet = writer.sheets[csv_name]
                worksheet.set_column('A:Z', 20, font_format)    
                # worksheet.set_row(0, 20, font_format) 
    elif mode == 'merge':
        df_list = []
        for csv_file in os.listdir(result_dir, ):
            if csv_file.endswith('.csv'):
                csv_name = get_file_name(csv_file)
                csv_path = os.path.join(result_dir, csv_file)
                df = pd.read_csv(csv_path, header=None)
                df.loc[-1] = [csv_name] + [np.nan] * (df.shape[1] - 1)
                df.index = df.index + 1
                df = df.sort_index()
                df_list.append(df)
        df = pd.concat(df_list, axis=0)
        df.columns = headers
        df.to_excel(writer, sheet_name=csv_dir, index=False)
        worksheet = writer.sheets[csv_dir]
        worksheet.set_column('A:Z', 20, font_format)  
        # worksheet.set_row(0, 20, font_format)



if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python <this.py> <device_result_dir>. e.g.: "python scripts/csv_to_xlsx.py result/amd9654"')
    csv_dir = os.path.join(sys.argv[1], 'summary')
    output_dir = csv_dir
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M')
    last_folder_name = os.path.basename(os.path.normpath(sys.argv[1]))
    excel_path = os.path.join(output_dir, f'{timestamp}_{last_folder_name}.xlsx')
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    headers = ['blas1, 每列：单个数组KB, 数据单位：median B/c', '4', '8', '16', '32', '64', '128', '256', '384', '512', '768', '1024', '1536', '2048', '3072', '4096', '8169', '16484','32768', '65536', '102400', '204800', '409600']
    csv_to_excel(writer, csv_dir, 'blas1', headers, mode='merge')
    headers = None
    csv_to_excel(writer, csv_dir, 'comp_comm', headers, mode='split')
    headers = ['Roofline, 每列：Compute/LD Ratio, 数据单位：95%tile flops/c', '1/4', '1/2', '1/1', '2/1', '4/1', '8/1']
    csv_to_excel(writer, csv_dir, 'roofline', headers, mode='merge')
    headers = ['stencil matrix 100 x ncols, 每列：矩阵一行的长度, 数据单位：median B/c', '100', '1024', '10240', '102400', '1024000']
    csv_to_excel(writer, csv_dir, 'stencil', headers, mode='split')
    writer.close()
    print(f'Excel file saved to {excel_path}')
