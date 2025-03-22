# Plot Module For TPBenchmark
# 确保在Jupyter Notebook中运行
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ipywidgets import widgets
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
import seaborn as sns

class Figure_Info:
    def __init__(self, title, xaxis_title, yaxis_title):
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title



def draw_fixed_heatmap(data_to_plot=pd.DataFrame, title_name='', save_dir = '', file_name=''):
    plt.figure(figsize=(20, 12))
    sns.heatmap(data_to_plot, annot=False, cmap='viridis')
    plt.title(title_name)
    plt.xlabel('Step')
    plt.ylabel('Rank')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    file_path = os.path.join(save_dir, f'{file_name}.png')
    plt.savefig(file_path)
    file_path = os.path.join(save_dir, f'{file_name}.svg')
    plt.savefig(file_path)       
    plt.close()



def mean_and_standard_error(values):
    mean_val = np.mean(values)
    std_dev = np.std(values, ddof=1)  # Use ddof=1 for sample standard deviation
    standard_error = std_dev / np.sqrt(len(values))
    return mean_val, standard_error

def remove_outliers(data, threshold=3):
    # IQR (Interquartile Range) method for outlier detection
    q25, q75 = np.percentile(data, 15), np.percentile(data, 85)
    iqr = q75 - q25
    cut_off = iqr * threshold
    lower, upper = q25 - cut_off, q75 + cut_off
    return data[(data >= lower) & (data <= upper)]

def draw_2D_histogram(data_to_plot=pd.DataFrame, title_name='', save_dir = '', file_name=''):
    # Drawing histograms
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))  # Create a figure and axes with 2x2 grid
    fig.suptitle(title_name, fontsize=16)

    stats = ['mean', 'min', 'max', '50%']
    titles = ['Means', 'Minimums', 'Maximums', '50th Percentiles']
    colors = ['blue', 'green', 'red', 'purple']

    for ax, stat, title, color in zip(axes.flatten(), stats, titles, colors):

        filtered_data = remove_outliers(data_to_plot[stat])
        # Calculate mean and standard error
        mean_val, standard_error = mean_and_standard_error(filtered_data)
        # Plot histogram
        ax.hist(filtered_data, bins=10, color=color, alpha=0.7)
        ax.set_title(f'Histogram of {title}\nMean: {mean_val:.4g}, Standard Error: {standard_error:.4g}')
        ax.set_xlabel(f'{title} Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplots to fit the figure area.
    
    # plt.show()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    file_path = os.path.join(save_dir, f'{file_name}.png')
    plt.savefig(file_path)
    file_path = os.path.join(save_dir, f'{file_name}.svg')
    plt.savefig(file_path)
    plt.close()

def draw_1D_histogram(data_to_plot=pd.DataFrame, title_name='', save_dir = '', file_name=''):
    # Drawing histograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, ax in enumerate(axes.flatten()):
        if i == 1: 
            data_to_plot['values'] = remove_outliers(data_to_plot['values'])
            title_name += ' (Outliers Removed)'

        ax.hist(data_to_plot['values'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)

        mean_val, standard_error = mean_and_standard_error(data_to_plot['values'])
        title_name += f'\nMean: {mean_val:.4g}, Standard Error: {standard_error:.4g}'

        # Calculate statistics
        # mean_val = np.mean(data_to_plot['values'])
        min_val = np.min(data_to_plot['values'])
        max_val = np.max(data_to_plot['values'])
        mid_val = np.median(data_to_plot['values'])

        # Plot vertical lines for statistics
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4g}')
        ax.axvline(min_val, color='green', linestyle='dashed', linewidth=1, label=f'Min: {min_val:.4g}')
        ax.axvline(max_val, color='blue', linestyle='dashed', linewidth=1, label=f'Max: {max_val:.4g}')
        ax.axvline(mid_val, color='purple', linestyle='dashed', linewidth=1, label=f'Median: {mid_val:.4g}')

        # Add legend
        ax.legend()

        # Set titles and labels
        ax.set_title(title_name)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')

    plt.tight_layout()

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    file_path = os.path.join(save_dir, f'{file_name}.png')
    plt.savefig(file_path)
    file_path = os.path.join(save_dir, f'{file_name}.svg')
    plt.savefig(file_path)
    plt.close()



def merge_two_figures(heatmap_filename:str = '', histogram_filename:str = '', save_dir:str = '', file_name:str = ''):
    fig, axes = plt.subplots(1, 2, figsize=(36, 12))
    img = plt.imread(heatmap_filename)
    axes[0].imshow(img)
    axes[0].axis('off')
    img = plt.imread(histogram_filename)
    axes[1].imshow(img)
    axes[1].axis('off')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name + '.png'))
    # plt.savefig(os.path.join(save_dir, file_name + '.svg'))
    plt.close()


def draw_statistic_scalability(data_to_plot:pd.DataFrame, title_name:str, save_dir:str, file_name:str):
    '''
    1 Figures:
        line chart of total_time_per_step(us)
        x-axis: N
        y-axis: commonly-total_time_per_step(us),compcomm-total_time_per_step(us),componly_GEMM-total_time_per_step(us)
    '''
    fig, axes = plt.subplots(figsize=(10, 8))
    y_colls = ['commonly-total_time_per_step(us)', 'compcomm-total_time_per_step(us)', 'componly_GEMM-total_time_per_step(us)']
    for y_col in y_colls:
        axes.plot(data_to_plot['N'], data_to_plot[y_col], label=y_col, marker='o')
    
    axes.set_xscale('log', base=2)
    axes.set_yscale('log', base=10)
    axes.set_xlabel('N')
    axes.set_ylabel('Time (us)')
    axes.set_title(title_name)
    axes.legend()

    
    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name + '.png')
    plt.savefig(save_path)
    plt.close(fig)


def __draw_single_distribution_best_mpi_func_alg(ax:plt.Axes, data_to_plot:pd.DataFrame, alg_col:str, x_values:list, y_values:list, z_value_col:str,title_name:str):
    colors_allreduce = {
        'fixed': 'blue', 'basicLinear': 'green', 'nonoverlapping': 'red', 'recursiveDoubling': 'purple', 'ring': 'orange', 'segmentedRing': 'brown', 'rabenseifner': 'pink'
    }
    colors_bcast = {
        'fixed': 'blue', 'basicLinear': 'green', 'chain': 'red', 'pipeline': 'purple', 'splitBinaryTree': 'orange', 'binaryTree': 'brown', 'binomial': 'pink', 'knomial': 'gray', 'scatterAllgather': 'white', 'scatterAllgatherRing': 'cyan'
    }

    tpkernel_name = data_to_plot['tpkernel_name'].iloc[0]
    colors = {}
    if 'allreduce' in tpkernel_name:
        colors = colors_allreduce 
    elif 'bcast' in tpkernel_name:
        colors = colors_bcast

    ax.set_title(title_name, fontsize=14)
    ax.set_xlabel('N', fontsize=12)
    ax.set_ylabel('Nr', fontsize=12)
    appeared_algs = set()
    for i, nr_val in enumerate(y_values):
        for j, n_val in enumerate(x_values):
            # 找到对应行的数据
            row = data_to_plot[(data_to_plot['N'] == n_val) & (data_to_plot['Nr'] == nr_val)]
            if len(row) == 1:
                row_data = row.iloc[0]
                best_alg = row_data[alg_col]
                val = row_data[z_value_col]
                # 获取颜色
                # c = colors.get(best_alg, '#cccccc')
                # 绘制方块
                # 将每个方块定位在(j, i)处，大小为1x1。Y轴朝上，因此需要倒置y轴或使用倒序的方式。
                # 这里让Nr对应y轴，N对应x轴，因此左上角为(0,0)，我们需要将上方行放高一点，可以将Y轴反转。
                rect = mpatches.Rectangle((j-0.5, i-0.5), 1, 1, facecolor=colors[best_alg], edgecolor='black')
                ax.add_patch(rect)

                # 在方块中央添加文字
                ax.text(j, i, f"{val:.4g}", ha='center', va='center', fontsize=16, color='black')
                appeared_algs.add(best_alg)
            else:
                # 如果没有对应数据或数据多余1条，可跳过或做特殊处理
                pass

    legend_handles = []
    for alg in appeared_algs:
        if alg in colors:
            legend_handles.append(mpatches.Patch(color=colors[alg], label=alg))
        else:
            # 对于不在colors字典内的算法，用默认颜色并说明
            legend_handles.append(mpatches.Patch(color='#cccccc', label=alg))

    # 设置坐标刻度
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(x) for x in x_values])
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(y) for y in y_values])
    ax.set_xlim(-0.5, len(x_values)-0.5)
    ax.set_ylim(-0.5, len(y_values)-0.5)
    ax.legend(handles=legend_handles)
    
    pass

def draw_distribution_best_mpi_func_alg(data_to_plot:pd.DataFrame, z_value_col:str, title_name:str, save_dir:str, file_name:str, mode='comm'):
    '''
    The headers of the data_to_plot should be:
        [
            'kernel_name', 'np', 'tpkernel_name', 'compute_unit', 'socket_num', 
            'data_size_comm(KiB)', 'data_size_GEMM(KiB)', 'data_size_memory(KiB)', 'ntest', 'N', 'Nr', 'processed_unit',
            'commonly-best_func_alg', 'commonly-comm-mean-adv', 'commonly-comm-std_err-adv', 'commonly-comm-max-adv', # best func alg is the algorithm whose total_time_per_step is the smallest
            'compcomm-best_func_alg', 'compcomm-comm-mean-adv', 'compcomm-comm-std_err-adv', 'compcomm-comm-max-adv', 
            'total_time_per_step(us)-theoretical-adv', 'total_time_per_step(us)-adv' #{compcomm(best)-componly_GEMM-commonly_comm(best)}/(componly_GEMM + commonly_comm(best))
        ]
    The figure is composed of two subfigures:
        fig1: commonly
            x-axis: N
            y-axis: Nr
            z-axis: color represents the best commonly-best_func_alg, text of value is the total_time_per_step(us)-theoretical-adv
        fig2: compcomm
            x-axis: N
            y-axis: Nr
            z-axis: color represents the best compcomm-best_func_alg, text of value is the total_time_per_step(us)-adv
    '''
    unique_N = sorted(data_to_plot['N'].unique())
    unique_Nr = sorted(data_to_plot['Nr'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for i, ax in enumerate(axes.flatten()):
        if mode == 'comm':
            if i == 0:
                program_pattern = 'commonly'
            else:
                program_pattern = 'compcomm'
            alg_col = f'{program_pattern}-best_func_alg'
            val_col = f'{program_pattern}-{z_value_col}'
            __draw_single_distribution_best_mpi_func_alg(ax, data_to_plot, x_values=unique_N, alg_col = alg_col,y_values=unique_Nr, z_value_col=val_col, title_name=f'{program_pattern} {val_col}\n{title_name}')
        elif mode == 'GEMM':
            alg_col = f'compcomm-best_func_alg'
            val_col = ''
            if i == 0:
                val_col = f'compcomm-comm-mean-adv'
            elif i == 1:
                val_col = f'compcomm-comm-max-adv'
            __draw_single_distribution_best_mpi_func_alg(ax, data_to_plot, x_values=unique_N, alg_col = alg_col,y_values=unique_Nr, z_value_col=val_col, title_name=f'compcomm {val_col}\n{title_name}')
        elif mode == 'GEMMcomm':
            alg_col = f'compcomm-best_func_alg'
            val_col = ''
            if i == 0:
                val_col = f'compcomm-GEMMcomm-mean-adv'
            elif i == 1:
                val_col = f'compcomm-GEMMcomm-max-adv'
            __draw_single_distribution_best_mpi_func_alg(ax, data_to_plot, x_values=unique_N, alg_col = alg_col,y_values=unique_Nr, z_value_col=val_col, title_name=f'compcomm {val_col}\n{title_name}')
        else:
            exit(f"Invalid mode: {mode}, please choose from ['comm', 'GEMM', 'GEMMcomm']")

    plt.tight_layout()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, file_name + '.png')
    plt.savefig(save_path)
    plt.close(fig)


def __draw_signle(data_to_plot:pd.DataFrame, x:str, title_name:str, ax:plt.Axes, processed_unit:str, x_log:int=0):
    data = data_to_plot[data_to_plot['processed_unit'] == processed_unit]
    ylabel = 'Time (us)' if processed_unit == 'us' else 'Bandwidth (KB/s)'
    if len(data[x]) != len(data['mean']):
        exit(f"Length of {x} and mean are not equal in {processed_unit}")
    ax.plot(data[x], data['mean'], label='mean', marker='o')

    ax2=ax.twinx()
    ax2.plot(data[x], data['standard_error'], label='standard_error', marker='x', color='black')
    ax2.legend(loc='upper right')
    ax2.set_ylabel('standard_error')

    if x != 'mpi_func_alg':
        ax.fill_between(data[x], data['min'], data['max'], color='gray', alpha=0.2, label='min/max')
        ax.set_xlabel(x)
    else:
        ax.plot(data[x], data['min'], label='min', marker='x')
        ax.plot(data[x], data['max'], label='max', marker='x')
        plt.setp(ax.get_xticklabels(), rotation=13, ha='center')
    
    if x_log >= 2:
        ax.set_xscale('log', base=x_log)
    ax.set_yscale('log')
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title_name} {processed_unit} Metrics')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.tight_layout()

def __draw_compare_CommonlyCompcomm(data_to_plot:pd.DataFrame, x:str, processed_unit:str, title_name:str, save_dir:str, file_name_base:str):
        def draw_single_compare_CommonlyCompcomm(x:str, ax:plt.Axes, y1:str, y2:str):
            colors = [['blue', 'green'], ['red', 'purple']]
            ax2 = ax.twinx()       
            for i, program_pattern in enumerate(['compcomm', 'commonly']): 
                data_subset = data[data['program_pattern'] == program_pattern]
                if data_subset.empty:
                    print(f"Empty data for {program_pattern}")
                    pass
                ax.plot(data_subset[x], data_subset[y1], label=f"{program_pattern}'s {y1}", color=colors[i][0], marker='o')
                ax2.plot(data_subset[x], data_subset[y2], label=f"{program_pattern}'s {y2}", color=colors[i][1], marker='x')
            if x != 'mpi_func_alg':
                ax.set_xlabel(x)
            else:
                plt.setp(ax.get_xticklabels(), rotation=13, ha='center')

            ax.set_yscale('log')
            ax.set_ylabel(ylabel + ' ' + y1)
            ax.legend(loc='upper left')
            ax.set_title(f"{title_name} {processed_unit} {y1} and {y2}")
            ax.grid(True)

            ax2.set_ylabel(y2)
            ax2.set_yscale('log')
            ax2.legend(loc='upper right')

        data = data_to_plot[data_to_plot['processed_unit'] == processed_unit]
        ylabel = 'Time (us)' if processed_unit == 'us' else 'Bandwidth (KB/s)' #KB_per_sec

        fig1, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        draw_single_compare_CommonlyCompcomm(x, axes[0], 'mean', 'standard_error')
        draw_single_compare_CommonlyCompcomm(x, axes[1], 'max', 'min')

        plt.tight_layout()
        file_name = f"{file_name_base}-{processed_unit}.png".replace(' ', '-')
        plt.savefig(os.path.join(save_dir, file_name))
        plt.close()

def draw_statistic_all_comm(data_to_plot:pd.DataFrame, x:str, title_name:str, save_dir:str, file_name_base:str):
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 8))
    x_log = 1
    if not data_to_plot[(data_to_plot['program_chunk'].str.contains('GEMM'))].empty:
        x_log = 2
    __draw_signle(data_to_plot = data_to_plot, x = x, title_name=title_name, ax=axes1[0], processed_unit='us', x_log=x_log)
    __draw_signle(data_to_plot = data_to_plot, x = x, title_name=title_name, ax=axes1[1], processed_unit='KB_per_sec', x_log=x_log)
    file_name=f"{file_name_base}.png".replace(' ', '-')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()



def draw_statistic_compare_CommonlyCompcomm(data_to_plot:pd.DataFrame, x:str, title_name:str, save_dir:str, file_name_base:str):
    # 确保 data_to_plot 非空
    if data_to_plot.empty:
        pass
    __draw_compare_CommonlyCompcomm(data_to_plot = data_to_plot, x=x, title_name = title_name, save_dir=save_dir, file_name_base=file_name_base, processed_unit = 'us')
    __draw_compare_CommonlyCompcomm(data_to_plot = data_to_plot, x=x, title_name = title_name, save_dir=save_dir, file_name_base=file_name_base, processed_unit = 'KB_per_sec')

    ############################################################



def __draw_compare_CommComp(data_to_plot:pd.DataFrame, x:str, processed_unit:str, title_name:str, save_dir:str, file_name_base:str):
    if not processed_unit in ['us', 'KB_per_sec']:
        exit(f"Invalid processed_unit: {processed_unit}, please choose from ['us', 'KB_per_sec']")
    if data_to_plot.empty:
        pass
    fig1, axes = plt.subplots(1, 2, figsize=(16, 8))
    data_comm = data_to_plot[
        (data_to_plot['program_chunk'].str.contains('comm'))
    ]
    data_gemm = data_to_plot[
        (data_to_plot['program_chunk'].str.contains('GEMM'))
    ]
    __draw_signle(data_to_plot = data_comm, x=x, title_name='COMM '+ title_name, ax=axes[0], processed_unit=processed_unit)
    __draw_signle(data_to_plot = data_gemm, x=x, title_name='GEMM '+ title_name, ax=axes[1], processed_unit=processed_unit)
    file_name=f"{file_name_base}-{processed_unit}.png".replace(' ', '-')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, file_name))
    plt.close()


def draw_statistic_compare_CommComp(data_to_plot:pd.DataFrame, x:str,title_name:str, save_dir:str, file_name_base:str):
    if data_to_plot.empty:
        pass
    __draw_compare_CommComp(data_to_plot = data_to_plot, x=x, processed_unit='us', title_name=title_name, save_dir=save_dir, file_name_base=file_name_base)
    __draw_compare_CommComp(data_to_plot = data_to_plot, x=x, processed_unit='KB_per_sec', title_name=title_name, save_dir=save_dir, file_name_base=file_name_base)



def draw_interactive_figures(data_to_plot = pd.DataFrame, title_name = '', file_name = '', save_dir = ''):
    '''
    Draw interactive heatmap and histogram for the given data

    Parameters
    ----------
    '''
    heatmap_data = data_to_plot.values
    histogram_data = data_to_plot.values.flatten()
    histogram_data_removed= remove_outliers(histogram_data, threshold=5)

    heatmap_title = f'Heatmap of {title_name}'
    histogram_title = f'Histogram of {title_name}'

    heatmap_file_name = f'Heatmap-of-{file_name}'
    histogram_file_name = f'Histogram-of-{file_name}'

    heatmap_dir = os.path.join(save_dir, 'heatmap')
    histogram_dir = os.path.join(save_dir, 'histogram')

    # Create heatmap

    heatmap_fig = go.FigureWidget(
        data=go.Heatmap(
            z=heatmap_data,
            colorscale='Viridis',
            hoverongaps=False,
            colorbar=dict(title='Value'),
            x=np.arange(heatmap_data.shape[1]),
            y=np.arange(heatmap_data.shape[0]),
        )
    )

    xaxis_font = 1000/(heatmap_data.shape[1])
    yaxis_font = 640/(heatmap_data.shape[0])

    heatmap_fig.update_layout(
        title={
            'text': heatmap_title, 
            'y':0.94,  # 可以调整标题在图中的垂直位置
            'x':0.5,  # 将标题x位置设为0.5,即图的中心
            'xanchor': 'center',  # 以x位置为中心点
            'yanchor': 'top'  # 以y位置为顶点
        },
        xaxis_title='Step', 
        yaxis_title='Rank', 
        width=1000, 
        height=800,
        xaxis=dict(
            tickfont=dict(
                size=xaxis_font,  # x轴刻度文字大小
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=yaxis_font,  # y轴刻度文字大小
            )
        )
    )

    ## 反转y轴以匹配NumPy数组的索引方向
    heatmap_fig.update_yaxes(
        autorange='reversed',  # 反转y轴
        tickmode='linear',
        dtick=1,
        
    )
    heatmap_fig.update_xaxes(tickmode='linear', dtick=1)


    # Create histogram
    ## 计算直方图的bin边界
    bin_num = 50
    counts, bin_edges = np.histogram(histogram_data, bins=bin_num)
    bin_size = (bin_edges[-1] - bin_edges[0]) / bin_num

    count_removed, bin_edges_removed = np.histogram(histogram_data_removed, bins=bin_num)
    bin_size_removed = (bin_edges_removed[-1] - bin_edges_removed[0]) / bin_num

    # if len(counts) > 25:
    #     counts, bin_edges = np.histogram(histogram_data, bins=25)
    #     bin_size = (bin_edges[1] - bin_edges[0]) / 25

    ## 计算统计量
    min_val = histogram_data.min()
    max_val = histogram_data.max()
    median_val = np.median(histogram_data)
    mean_val, standard_error = mean_and_standard_error(histogram_data)
    histogram_title += f'<br>Mean: {mean_val:.4g}, Standard Error: {standard_error:.4g}'

    ## 添加统计量的垂直线作为 Scatter trace
    def add_stat_traces(fig, stats):
        """
        在直方图上添加统计量的垂直线作为 Scatter trace。

        参数：
        - fig: Plotly FigureWidget 对象。
        - stats: 字典，包含统计量名称和对应的值。
        """
        colors = {
            'Min': 'green',
            'Max': 'red',
            'Median': 'purple',
            'Mean': 'orange'
        }
        dash_styles = {
            'Min': 'dash',
            'Max': 'dash',
            'Median': 'dot',
            'Mean': 'dot'
        }
        for stat, value in stats.items():
            trace = go.Scatter(
                x=[value, value],
                y=[0, max(counts) + 5],
                mode='lines',
                line=dict(color=colors[stat], width=3, dash=dash_styles[stat]),
                name=f"{stat}: {value:.4g}"
            )
            fig.add_trace(trace)

    
    stats = {
        'Min': min_val,
        'Max': max_val,
        'Median': median_val,
        'Mean': mean_val
    }

    hist_fig_data = go.Histogram(
                x=histogram_data,
                xbins=dict(
                    start=bin_edges[0],
                    end=bin_edges[-1],
                    size=bin_size
                ),
                marker_color='blue',
                autobinx=False
            )


    hist_fig_data_removed = go.Histogram(
                x=histogram_data_removed,
                xbins=dict(
                    start=bin_edges_removed[0],
                    end=bin_edges_removed[-1],
                    size=bin_size_removed
                ),
                marker_color='blue',
                autobinx=False
            )

    
    hist_fig = go.FigureWidget(
        data=hist_fig_data,
    )
    hist_fig.update_layout(
        title={
            'text': histogram_title, 
            'y':0.96,  # 可以调整标题在图中的垂直位置
            'x':0.5,  # 将标题x位置设为0.5,即图的中心
            'xanchor': 'center',  # 以x位置为中心点
            'yanchor': 'top'  # 以y位置为顶点
        }, 
        width=1000, 
        height=800,
        xaxis=dict(
            tickfont=dict(
                size=10,  # x轴刻度文字大小
            )
        ),
        yaxis=dict(
            tickfont=dict(
                size=10,  # y轴刻度文字大小
            )
        )
    )
    add_stat_traces(hist_fig, stats)
    hist_fig.update_layout(
        legend=dict(
            orientation='h',        # 图例水平排列
            yanchor='bottom',       # 图例的锚点在其底部
            y=0.96,                 # 将图例的底部锚点设置在绘图区上方一点点（超过1表示在绘图区上方）
            xanchor='center',       # 锚点在图例的中间位置
            x=0.5                   # 居中放置图例（0.5表示图幅的中点）
        ),
        margin=dict(t=100)          # 增加顶部边距，为图例腾出空间
    )

    # 创建一个文本区域用于显示信息
    info_box = widgets.Textarea(value='', layout={'width': '95%', 'height': '60px'})

    # 显示图形和信息框
    reset_button = widgets.Button(description='重置')
    save_button = widgets.Button(description='保存')
    remove_outliers_button = widgets.Button(description='移除离群值')
    hbox = widgets.HBox([heatmap_fig, hist_fig], layout=widgets.Layout(
        width='100%',
        overflow_x='auto',
        flex_flow='row wrap'
    ))
    container = widgets.VBox([hbox, widgets.HBox([reset_button, save_button, remove_outliers_button]), info_box])
    display(container)

    # 定义heatmap的点击事件处理函数
    def on_heatmap_click(trace, points, selector):
        if points.point_inds:
            # 获取点击的x和y索引
            x_index = points.xs[0]
            y_index = points.ys[0]
            value = heatmap_data[y_index][x_index]

            # 更新histogram
            with hist_fig.batch_update():
                # 移除之前的形状（如果有）
                hist_fig.data = [trace for trace in hist_fig.data if 'Selected Value' not in trace.name]
                if value in hist_fig.data[0].x:
                    # 添加一条竖线表示选中的值
                    selected_trace = go.Scatter(
                        x=[value, value], 
                        y=[0, max(counts) + 5],
                        mode='lines',
                        line=dict(color='black', width=3, dash='dash'),
                        name=f'Selected Value: {value:.4g}'
                    )
                    hist_fig.add_trace(selected_trace)
            info_box.value = f"在histogram中选中的heatmap值：{value:.4f}，位置：({y_index}, {x_index})\n"

    # 定义histogram的点击事件处理函数
    def on_hist_click(trace, points, selector):
        if points.xs:
            # 获取点击的x坐标
            x_clicked = points.xs[0]
            # 根据x_clicked找到对应的bin_index
            bin_index = np.digitize(x_clicked, bin_edges) - 1
            # 防止bin_index越界
            bin_index = max(min(bin_index, len(bin_edges) - 2), 0)
            bin_left = bin_edges[bin_index]
            bin_right = bin_edges[bin_index + 1]
            # 获取该bin内的数据点索引
            indices = np.argwhere((heatmap_data >= bin_left) & (heatmap_data < bin_right))
            # 更新heatmap，框住这些数据点
            with heatmap_fig.batch_update():
                # 移除之前的形状（如果有）
                shapes = []
                if len(indices) > 0:
                    for idx in indices:
                        y_index, x_index = idx
                        # Heatmap方块的边界
                        shape = dict(
                            type='rect',
                            x0=x_index - 0.5,
                            x1=x_index + 0.5,
                            y0=y_index - 0.5,
                            y1=y_index + 0.5,
                            xref='x',
                            yref='y',
                            line=dict(color='red', width=2),
                            fillcolor='rgba(0,0,0,0)',
                            layer='above'
                        )
                        shapes.append(shape)
                # 更新形状
                if 'shapes' in heatmap_fig.layout and heatmap_fig.layout.shapes:
                    heatmap_fig.layout.shapes += tuple(shapes)
                else:
                    heatmap_fig.layout.shapes = shapes
            # 更新信息框
            info_box.value = f"选中的histogram区间：[{bin_left:.4f}, {bin_right:.4f})\n已在heatmap上框住该区间内的所有数据点。"

    # 定义重置按钮的回调函数
    def on_reset_button_clicked(b):
        with hist_fig.batch_update():
            hist_fig.data[0].x = histogram_data
            hist_fig.data[0].xbins=dict(
                    start=bin_edges[0],
                    end=bin_edges[-1],
                    size=bin_size
                )
            hist_fig.data = [trace for trace in hist_fig.data if 'Selected Value' not in trace.name]
            # add_stat_traces(hist_fig, stats)
        with heatmap_fig.batch_update():
            heatmap_fig.layout.shapes = []
        info_box.value = '已重置heatmap和histogram'

    def on_save_button_clicked(b):
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)
        if not os.path.exists(histogram_dir):
            os.makedirs(histogram_dir)
        heatmap_modified_flag=''
        histogram_modified_flag=''
        if heatmap_fig.layout.shapes == []:
            heatmap_modified_flag = '-modified'
        if hist_fig.layout.shapes == []:
            histogram_modified_flag = '-modified'
        
        heatmap_fig.write_image(os.path.join(heatmap_dir, f'{heatmap_file_name}{heatmap_modified_flag}.svg'))
        hist_fig.write_image(os.path.join(histogram_dir, f'{histogram_file_name}{histogram_modified_flag}.svg'))
        info_box.value = f'已保存heatmap和histogram至'

    def on_remove_outliers_clicked(b):
        with hist_fig.batch_update():
            hist_fig.data[0].x = histogram_data_removed
            hist_fig.data[0].xbins=dict(
                    start=bin_edges_removed[0],
                    end=bin_edges_removed[-1],
                    size=bin_size_removed
                )
            hist_fig.data = [trace for trace in hist_fig.data if 'Selected Value' not in trace.name]
        info_box.value = '已移除histogram中的离群值'

    # 绑定事件
    heatmap_fig.data[0].on_click(on_heatmap_click)
    hist_fig.data[0].on_click(on_hist_click)
    reset_button.on_click(on_reset_button_clicked)
    save_button.on_click(on_save_button_clicked)
    remove_outliers_button.on_click(on_remove_outliers_clicked)
