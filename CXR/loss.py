import torch
import torch.nn.functional as F
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import pandas as pd
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
from torch import nn

def torch_sobel_edge(mask: torch.Tensor, sigma=1.0) -> torch.Tensor:
    """GPU加速的Sobel边缘检测，输出[0,1]边缘概率"""
    # 高斯平滑
    kernel_size = int(4*sigma + 1)
    mask_float = mask.float()
    blurred = T.functional.gaussian_blur(mask_float, kernel_size=[kernel_size]*2, sigma=[sigma]*2)
    
    # Sobel算子
    sobel_x = F.conv2d(blurred, torch.tensor([[[[-1,0,1], [-2,0,2], [-1,0,1]]]], 
                        device=mask.device, dtype=blurred.dtype), padding=1)
    sobel_y = F.conv2d(blurred, torch.tensor([[[[-1,-2,-1], [0,0,0], [1,2,1]]]], 
                        device=mask.device, dtype=blurred.dtype), padding=1)
    edge_mag = torch.sqrt(sobel_x**2 + sobel_y**2)
    return (edge_mag > 0.1 * edge_mag.max()).float()  # 二值化

class MixedEdgeWeightedLoss(nn.Module):
    def __init__(self, edge_scale=5.0, sigma=1.0):
        super().__init__()
        self.edge_scale = edge_scale
        self.sigma = sigma  # 控制边缘检测的平滑度

    def forward(self, gt, pred, prd_scores):
        edge_map = torch_sobel_edge(gt.unsqueeze(1), self.sigma)  # [B,1,H,W]
        weight_map = 1 + (self.edge_scale - 1) * edge_map
        edge_weight = 1.0 + 2.0 * edge_map # 边缘区域权重设为5
        weighted_seg_loss = (- (edge_weight * gt) * torch.log(pred + 1e-6) 
                             - (edge_weight * (1 - gt)) * torch.log((1 - pred) + 1e-6)).mean()

        inter = (gt * (pred > 0.5)).sum(1).sum(1)
        iou = (inter / (gt.sum((1, 2)) + (pred > 0.5).sum((1, 2)) - inter)).mean()
        iou_loss = (1 - iou)
        dice = ((2 * iou) / (1 + iou)).mean()
        dice_loss = (1 - dice)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = weighted_seg_loss + (score_loss + iou_loss + dice_loss) * 0.05 # mix losses
        return loss, iou, dice





