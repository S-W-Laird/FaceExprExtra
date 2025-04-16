import timm
import torch.nn as nn
from safetensors.torch import load_file

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name='vit_large_patch16_224', ckpt_path=None):
        super(ViTFeatureExtractor, self).__init__()
        # self.vit = timm.create_model(model_name, pretrained=False, num_classes=0)  # 不自动下载
        self.vit = timm.create_model(model_name, pretrained=True)
        if ckpt_path is not None:
            state_dict = load_file(ckpt_path)  # 用 safetensors 读取
            self.vit.load_state_dict(state_dict, strict=False)
        
        for param in self.vit.parameters():
            param.requires_grad = False     # 冻结所有参数
            
    def forward(self, x):
        return self.vit(x)