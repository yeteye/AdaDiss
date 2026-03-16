import os
import threading

import spatialdata as sd
from spatialdata_io import xenium


def xenium_data_load_multithreaded(data_dir, sample_info):
    """
    多线程读取多个 Xenium 样本数据，合并为一个 SpatialData 对象，
    并添加样本分组信息及细胞边界注释。

    Parameters
    ----------
    data_dir : str
        包含各样本原始数据的根目录路径。
    sample_info : str
        样本信息文件路径，文件为制表符分隔的文本，每行包含：
        raw_name（原始文件夹名）、sample_name（样本名）、group_name（分组名）
        可根据实际情况调整列数，函数只取前三列。

    Returns
    -------
    sdata : SpatialData
        合并后的 SpatialData 对象，包含所有样本的细胞表、空间图像和形状。
    """

    def sd_read_xenium(sample_data, sample_name, sdata_dict):
        """读取单个样本的 Xenium 数据，存入共享字典"""
        sdata = xenium(
            path=sample_data,
            cells_boundaries=True,
            n_jobs=6
        )
        sdata_dict[sample_name] = sdata

    threads = []
    sdata_dict = {}
    sample_2_group = {}

    # 读取样本信息文件，启动每个样本的读取线程
    with open(sample_info, 'r') as f:
        for line in f:
            raw_name, sample_name, group_name = line.strip().split('\t')[:3]
            sample_2_group[sample_name] = group_name

            thread = threading.Thread(
                target=sd_read_xenium,
                args=(os.path.join(data_dir, raw_name), sample_name, sdata_dict)
            )
            threads.append(thread)
            thread.start()

    # 等待所有线程完成
    for thread in threads:
        thread.join()

    # 合并所有样本的 SpatialData 对象
    sdata = sd.concatenate(
        sdata_dict,
        concatenate_tables=True,          # 合并表数据（单细胞表达矩阵）
        obs_names_make_unique=True         # 确保观察名称唯一
    )

    # 在合并后的表（'table'）中添加样本信息
    table = sdata.tables['table']
    # 从 'region' 列提取样本名（格式：'cell_circles-样本名'）
    table.obs["sample"] = table.obs["region"].str.replace('cell_circles-', '')
    # 根据样本名映射分组
    table.obs["group"] = table.obs["sample"].apply(lambda x: sample_2_group[x])
    # 创建细胞边界对应的区域列
    table.obs["cell_boundaries"] = table.obs["region"].str.replace('cell_circles', 'cell_boundaries')

    # 将表与细胞边界形状关联
    cell_boundary_keys = [key for key in sdata.shapes.keys() if key.startswith('cell_boundaries-')]
    sdata.set_table_annotates_spatialelement(
        table_name='table',
        region=cell_boundary_keys,
        region_key='cell_boundaries'
    )

    return sdata