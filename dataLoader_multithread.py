import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import spatialdata as sd
from spatialdata_io import xenium
import pandas as pd

def xenium_data_load_multithreaded(data_dir, sample_info, max_workers=4, n_jobs_per_sample=4):
    """
    优化的多线程读取多个 Xenium 样本数据
    
    Parameters
    ----------
    data_dir : str
        包含各样本原始数据的根目录路径。
    sample_info : str
        样本信息文件路径。
    max_workers : int
        最大并发线程数，默认4。
    n_jobs_per_sample : int
        每个样本内部的并行数，默认4。
    
    Returns
    -------
    sdata : SpatialData
        合并后的 SpatialData 对象。
    """
    
    def read_single_sample(sample_data, sample_name, group_name):
        """读取单个样本，包含错误处理"""
        try:
            print(f"开始读取样本: {sample_name}")
            sdata = xenium(
                path=sample_data,
                cells_boundaries=True,
                n_jobs=n_jobs_per_sample
            )
            print(f"✅ 完成读取样本: {sample_name}")
            return {
                'sample_name': sample_name,
                'group_name': group_name,
                'sdata': sdata
            }
        except Exception as e:
            print(f"❌ 读取样本 {sample_name} 失败: {e}")
            return {
                'sample_name': sample_name,
                'group_name': group_name,
                'sdata': None,
                'error': str(e)
            }
    
    # 读取样本信息
    samples = []
    with open(sample_info, 'r') as f:
        for line in f:
            raw_name, sample_name, group_name = line.strip().split('\t')[:3]
            samples.append({
                'raw_name': raw_name,
                'sample_name': sample_name,
                'group_name': group_name,
                'full_path': os.path.join(data_dir, raw_name)
            })
    
    print(f"共发现 {len(samples)} 个样本")
    
    # 使用线程池并行读取
    sdata_dict = {}
    sample_2_group = {}
    failed_samples = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sample = {
            executor.submit(
                read_single_sample, 
                sample['full_path'], 
                sample['sample_name'],
                sample['group_name']
            ): sample for sample in samples
        }
        
        # 收集结果
        for future in as_completed(future_to_sample):
            sample = future_to_sample[future]
            try:
                result = future.result(timeout=300)  # 5分钟超时
                if result['sdata'] is not None:
                    sdata_dict[result['sample_name']] = result['sdata']
                    sample_2_group[result['sample_name']] = result['group_name']
                else:
                    failed_samples.append((sample['sample_name'], result.get('error', 'Unknown error')))
            except Exception as e:
                failed_samples.append((sample['sample_name'], str(e)))
                print(f"处理样本 {sample['sample_name']} 时发生错误: {e}")
    
    # 报告失败的样本
    if failed_samples:
        print(f"⚠️ 以下样本读取失败:")
        for sample_name, error in failed_samples:
            print(f"  - {sample_name}: {error}")
    
    if not sdata_dict:
        raise ValueError("没有成功读取任何样本")
    
    print(f"成功读取 {len(sdata_dict)}/{len(samples)} 个样本")
    
    # 合并所有样本
    print("正在合并样本...")
    sdata = sd.concatenate(
        sdata_dict,
        concatenate_tables=True,
        obs_names_make_unique=True
    )
    
    # 添加元数据
    table = sdata.tables['table']
    
    # 从 region 列提取样本名
    table.obs["sample"] = table.obs["region"].str.replace('cell_circles-', '')
    
    # 映射分组
    table.obs["group"] = table.obs["sample"].map(sample_2_group)
    
    # 处理可能的缺失值
    if table.obs["group"].isna().any():
        print("⚠️ 部分样本的分组信息缺失，使用 'unknown' 填充")
        table.obs["group"] = table.obs["group"].fillna('unknown')
    
    # 创建细胞边界对应的区域列
    table.obs["cell_boundaries"] = table.obs["region"].str.replace('cell_circles', 'cell_boundaries')
    
    # 关联细胞边界形状
    cell_boundary_keys = [key for key in sdata.shapes.keys() if key.startswith('cell_boundaries-')]
    
    if cell_boundary_keys:
        sdata.set_table_annotates_spatialelement(
            table_name='table',
            region=cell_boundary_keys,
            region_key='cell_boundaries'
        )
        print(f"✅ 成功关联 {len(cell_boundary_keys)} 个细胞边界")
    else:
        print("⚠️ 未找到细胞边界形状")
    
    print(f"✅ 数据处理完成，共 {len(table.obs)} 个细胞")
    
    return sdata


# 如果需要更高级的进度显示，可以使用这个版本
def xenium_data_load_with_progress(data_dir, sample_info, max_workers=4):
    """
    带进度条的多线程版本（需要安装 tqdm: pip install tqdm）
    """
    from tqdm import tqdm
    
    # 读取样本信息
    samples = []
    with open(sample_info, 'r') as f:
        for line in f:
            raw_name, sample_name, group_name = line.strip().split('\t')[:3]
            samples.append({
                'raw_name': raw_name,
                'sample_name': sample_name,
                'group_name': group_name,
                'full_path': os.path.join(data_dir, raw_name)
            })
    
    sdata_dict = {}
    sample_2_group = {}
    
    def read_with_progress(sample):
        try:
            sdata = xenium(
                path=sample['full_path'],
                cells_boundaries=True,
                n_jobs=4
            )
            return sample['sample_name'], sample['group_name'], sdata
        except Exception as e:
            print(f"样本 {sample['sample_name']} 失败: {e}")
            return sample['sample_name'], sample['group_name'], None
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(read_with_progress, sample) for sample in samples]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="读取样本"):
            sample_name, group_name, sdata = future.result()
            if sdata is not None:
                sdata_dict[sample_name] = sdata
                sample_2_group[sample_name] = group_name
    
    # 合并和处理（同上）
    sdata = sd.concatenate(sdata_dict, concatenate_tables=True, obs_names_make_unique=True)
    
    table = sdata.tables['table']
    table.obs["sample"] = table.obs["region"].str.replace('cell_circles-', '')
    table.obs["group"] = table.obs["sample"].map(sample_2_group)
    table.obs["cell_boundaries"] = table.obs["region"].str.replace('cell_circles', 'cell_boundaries')
    
    cell_boundary_keys = [key for key in sdata.shapes.keys() if key.startswith('cell_boundaries-')]
    if cell_boundary_keys:
        sdata.set_table_annotates_spatialelement(
            table_name='table',
            region=cell_boundary_keys,
            region_key='cell_boundaries'
        )
    
    return sdata