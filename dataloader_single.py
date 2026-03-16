import os
import spatialdata as sd
from spatialdata_io import xenium

def xenium_data_load_single_sample(data_dir, raw_name, sample_name, group_name):
    """
    直接读取单个 Xenium 样本，并添加元数据。
    
    Parameters
    ----------
    data_dir : str
        包含样本数据的根目录。
    raw_name : str
        样本原始文件夹名（位于 data_dir 下）。
    sample_name : str
        自定义样本名（将添加到 obs['sample']）。
    group_name : str
        样本分组名（将添加到 obs['group']）。
    """
    # 读取单个样本
    sdata = xenium(
        path=os.path.join(data_dir, raw_name),
        cells_boundaries=True,
        n_jobs=6
    )

    # 添加样本信息（无需合并，直接操作当前 sdata 的表）
    table = sdata.tables['table']
    table.obs["sample"] = sample_name
    table.obs["group"] = group_name

    # 生成细胞边界列（基于原有的 region 列）
    table.obs["cell_boundaries"] = table.obs["region"].str.replace('cell_circles', 'cell_boundaries')

    # 关联细胞边界形状
    cell_boundary_keys = [key for key in sdata.shapes.keys() if key.startswith('cell_boundaries-')]
    sdata.set_table_annotates_spatialelement(
        table_name='table',
        region=cell_boundary_keys,
        region_key='cell_boundaries'
    )

    return sdata