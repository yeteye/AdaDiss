import os
import warnings
import spatialdata as sd
from spatialdata_io import xenium

warnings.filterwarnings("ignore")

def xenium_data_load_single_sample(data_dir, raw_name, sample_name, group_name):
    """
    直接读取单个 Xenium 样本，并添加元数据。
    """
    # 读取单个样本
    sdata = xenium(
        path=os.path.join(data_dir, raw_name),
        cells_boundaries=True,
        n_jobs=6
    )
    
    # 添加样本信息
    table = sdata.tables['table']
    table.obs["sample"] = sample_name
    table.obs["group"] = group_name
    
    # 对于单样本，直接使用 'cell_boundaries' 作为区域标识
    if 'cell_boundaries' in sdata.shapes:
        table.obs["cell_boundaries"] = 'cell_boundaries'
        sdata.set_table_annotates_spatialelement(
            table_name='table',
            region='cell_boundaries',
            region_key='cell_boundaries'
        )
    
    return sdata