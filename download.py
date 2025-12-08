import torch
from depth_anything_3.api import DepthAnything3

# 加载支持3DGS的模型
# model_name 应该是 'da3-giant' 或 'da3nested-giant-large'
# model = DepthAnything3(model_name='da3-giant', load_from="local", local_path='/path/to/your/model.pth')
# 或者从云端自动下载
model = DepthAnything3(model_name='da3-giant')