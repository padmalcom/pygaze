import timm
import torch
from omegaconf import DictConfig

def create_model(config: DictConfig) -> torch.nn.Module:
    mode = config.mode
    model = timm.create_model(config.model.name, num_classes=2)
    device = torch.device(config.device)
    model.to(device)
    return model