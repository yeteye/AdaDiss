export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
conda activate spatial_gnn
python3 -c "
import torch
print('CUDA:', torch.cuda.get_device_name(0))
print('总显存:', torch.cuda.get_device_properties(0).total_memory / 1024**3, 'GB')

x = torch.zeros(20 * 1024**3 // 4, dtype=torch.float32, device='cuda')
print('✅ 成功分配 20GB')
del x
torch.cuda.empty_cache()

y = torch.zeros(40 * 1024**3 // 4, dtype=torch.float32, device='cuda')
print('✅ 成功分配 40GB')
del y
torch.cuda.empty_cache()
"