'''
Author: lichao951787328 951787328@qq.com
Date: 2025-12-31 14:52:20
LastEditors: lichao951787328 951787328@qq.com
LastEditTime: 2026-01-19 11:17:00
FilePath: /viplanner/viplanner/plannernet/rgb_encoder.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import pickle
from typing import Optional

import torch
import torch.nn as nn

# detectron2 and mask2former (used to load pre-trained models from Mask2Former)
# === 尝试导入 Detectron2 和 Mask2Former ===
# 这些是用于加载特定预训练模型配置的库
try:
    from detectron2.config import get_cfg
    from detectron2.modeling.backbone import build_resnet_backbone
    from detectron2.projects.deeplab import add_deeplab_config

    PRE_TRAIN_POSSIBLE = True # 标记导入成功
except ImportError:
    PRE_TRAIN_POSSIBLE = False # 导入失败，无法使用此功能
    print("[Warning] Pre-trained ResNet50 models cannot be used since detectron2" " not found")

try:
    from omni.viplanner.third_party.mask2former.mask2former import add_maskformer2_config
except ImportError:
    PRE_TRAIN_POSSIBLE = False
    print("[Warning] Pre-trained ResNet50 models cannot be used since" " mask2former not found")

# 获取 Mask2Former 的配置节点
def get_m2f_cfg(cfg_path: str):  # -> CfgNode:
    # load config from file
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(cfg_path)
    cfg.freeze()
    return cfg

# === RGB 编码器类 ===
class RGBEncoder(nn.Module):
    def __init__(self, cfg, weight_path: Optional[str] = None, freeze: bool = True) -> None:
        super().__init__()

        # load pre-trained resnet
        # 1. 构建 ResNet Backbone
        # 使用 detectron2 的接口构建，输入通道设为3
        input_shape = argparse.Namespace()
        input_shape.channels = 3
        self.backbone = build_resnet_backbone(cfg, input_shape)

        # load weights # 2. 加载预训练权重 (pickle 格式)
        if weight_path is not None:
            with open(weight_path, "rb") as file:
                model_file = pickle.load(file, encoding="latin1")
                
            # 调整权重字典的 key，去掉 "backbone." 前缀以匹配当前模型
            model_file["model"] = {k.replace("backbone.", ""): torch.tensor(v) for k, v in model_file["model"].items()}
            # 加载权重，允许非严格匹配 (strict=False)
            missing_keys, unexpected_keys = self.backbone.load_state_dict(model_file["model"], strict=False)
            if len(missing_keys) != 0:
                print(f"[WARNING] Missing keys: {missing_keys}")
                print(f"[WARNING] Unexpected keys: {unexpected_keys}")
            print(f"[INFO] Loaded pre-trained backbone from {weight_path}")

        # freeze network
        # 3. 冻结参数
        # 如果 freeze=True，则锁定 backbone 参数不参与梯度更新 (迁移学习常用做法)
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # layers to get correct output shape --> modifiable
        # 4. 降维层
        # ResNet50/101 的输出通常是 2048 通道，为了与 Decoder (输入512) 或 PlannerNet 对齐
        # 这里用一个卷积层将 2048 -> 512 通道
        self.conv1 = nn.Conv2d(2048, 512, kernel_size=3, stride=1, padding=1)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 获取 ResNet 的 'res5' 阶段输出 (Feature Map)
        # 输出尺寸通常是 H/32, W/32，通道 2048
        x = self.backbone(x)["res5"]  # size = (N, 2048, 12, 20) (height and width same as ResNet18)
        # 降维到 512 通道
        x = self.conv1(x)  # size = (N, 512,  12, 20)
        return x


# EoF
