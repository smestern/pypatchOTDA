

from .resnet import ResBase
from .classifier import CLS, ProtoCLS, Projection, ProtoNormCLS
from .vit import vit_base, vit_base_dino, deit_base
from .ff_net import FFNet

from torch import nn

CLIP_MODELS = ['']#clip.available_models()
DINOv2_MODELS = ['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14']
backbone_names = ['resnet18', 'resnet50'] + CLIP_MODELS + DINOv2_MODELS + ['vit_base', 'vit_base_dino', 'deit_base']
head_names = ['linear', 'mlp', 'prototype', 'protonorm']


def build_backbone(name):
    """
    build the backbone for feature exatraction
    """
    if 'resnet' in name:
        return ResBase(option=name)
    elif 'FF' in name:
        in_dim = int(name.split('_')[-2])
        out_dim = int(name.split('_')[-1])
        return FFNet(in_dim, out_dim)
    elif "IDENT" in name:
        in_dim = int(name.split('_')[-2])
        
        out_dim = int(name.split('_')[-1])
        return IDENT(in_dim, in_dim)
    elif name in CLIP_MODELS:
        pass#model, _ = clip.load(name)
    elif name in DINOv2_MODELS:
        import torch
        model = torch.hub.load('facebookresearch/dinov2', name)
        # model = torch.hub.load('/data1/deng.bin/coding/JUSTforLearning/dinov2', name, source='local')
    elif name == 'vit_base':
        model = vit_base(pretrained=False)
    elif name == 'vit_base_dino':
        model = vit_base_dino()
    elif name == 'deit_base':
        model = deit_base()
    else:
        raise RuntimeError(f"Model {name} not found; available models = {backbone_names}")
    
    return model.float()


def build_head(name, in_dim, out_dim, hidden_dim=2048, temp=0.05):
    if name == 'linear':
        return CLS(in_dim, out_dim)
    elif name == 'mlp':
        return Projection(in_dim, feat_dim=out_dim, hidden_mlp=hidden_dim)
    elif name == 'prototype':
        return ProtoCLS(in_dim, out_dim, temp=temp)
    elif name == 'protonorm':
        return ProtoNormCLS(in_dim, out_dim, temp=temp)
    else:
        raise RuntimeError(f"Model {name} not found; available models = {head_names}")
    

class IDENT(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(IDENT, self).__init__()
        self.input_dim = in_dim
        self.output_dim = out_dim
        
    def forward(self, x):
        return x