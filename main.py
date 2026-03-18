from dataLoader_multithread import xenium_data_load_multithreaded
from dataloader_single import xenium_data_load_single_sample
data_dir = "/home/ailab/caohao/"
raw_name = "Xenium_Prime_Human_Prostate_FFPE_outs"
sample_name = "Human_Prostate"
group_name = "First"
data = xenium_data_load_single_sample(data_dir, raw_name,sample_name,group_name)
