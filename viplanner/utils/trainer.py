# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

# python
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import tqdm
import wandb  # logging
import yaml

# imperative-planning-learning (项目内部模块)
from omni.viplanner.config import TrainCfg  # 训练配置类
from omni.viplanner.plannernet import (
    PRE_TRAIN_POSSIBLE,  # 预训练模型是否可用标志 是否具备使用特定第三方预训练模型（如 Mask2Former）的能力
    AutoEncoder,  # 基础编码器-解码器模型（可能用于深度图输入）
    DualAutoEncoder,  # 双输入编码器-解码器模型（可能用于深度+语义/RGB输入）
    get_m2f_cfg,  # 获取Mask2Former配置的函数
)
from omni.viplanner.traj_cost_opt import TrajCost, TrajViz  # 轨迹成本计算和轨迹可视化类
from omni.viplanner.utils.torchutil import EarlyStopScheduler, count_parameters  # 早期停止调度器和参数计数工具

from .dataset import PlannerData, PlannerDataGenerator  # 数据集和数据生成器类

torch.set_default_dtype(torch.float32)  # 设置PyTorch默认浮点类型为float32


class Trainer:
    """
    VIPlanner Trainer
    """

    def __init__(self, cfg: TrainCfg) -> None:
        self._cfg = cfg

        # set model save/load path
        os.makedirs(self._cfg.curr_model_dir, exist_ok=True)  # 创建当前模型保存目录
        self.model_path = os.path.join(self._cfg.curr_model_dir, "model.pt")  # 设置模型保存的完整路径
        if self._cfg.hierarchical:  # 如果配置了分层训练
            self.model_dir_hierarch = os.path.join(self._cfg.curr_model_dir, "hierarchical")
            os.makedirs(self.model_dir_hierarch, exist_ok=True)
            self.hierach_losses = {}

        # image transforms
        self.transform = transforms.Compose(  # 定义图像预处理管道，之后输出的图像都这个来处理
            [
                # ToTensor() 操作会自动将数据除以 255（如果是 uint8 类型）或者直接保留（如果是 float）
                transforms.ToTensor(),  # 将PIL图像或numpy数组转换为PyTorch张量
                transforms.Resize((self._cfg.img_input_size), antialias=True),  # 调整图像大小到指定输入尺寸
            ]
        )

        # init buffers DATA (数据相关缓冲区)
        self.data_generators: List[PlannerDataGenerator] = []  # 数据生成器列表，每个环境一个
        self.data_traj_cost: List[TrajCost] = []  # 轨迹成本计算对象列表, 每个环境一个
        self.data_traj_viz: List[TrajViz] = []  # 轨迹可视化对象列表, 每个环境一个
        self.fov_ratio: float = None  # 视场角比例，用于分层训练中的数据采样
        self.front_ratio: float = None  # 前方视场比例，用于分层训练中的数据采样
        self.back_ratio: float = None  # 后方视场比例，用于分层训练中的数据采样
        self.pixel_mean: np.ndarray = None  # 图像像素均值，用于归一化
        self.pixel_std: np.ndarray = None  # 图像像素标准差，用于归一化

        # inti buffers MODEL (模型相关缓冲区)
        self.best_loss = float("inf")  # 记录最佳验证损失，初始化为无穷大
        self.test_loss = float("inf")  # 记录测试损失，初始化为无穷大
        self.net: nn.Module = None  # 神经网络模型实例 _load_model中进行初始化
        self.optimizer: optim.Optimizer = None  # 优化器实例
        self.scheduler: EarlyStopScheduler = None  # 早期停止调度器实例

        print("[INFO] Trainer initialized")
        return

    """PUBLIC METHODS"""

    def train(self) -> None:
        print("[INFO] Start Training")
        # init logging
        self._init_logging()
        # load model and prepare model for training
        self._load_model(self._cfg.resume)  # 加载模型，如果cfg.resume为True则从检查点恢复
        self._configure_optimizer()  # 配置优化器和学习率调度器

        # get dataloader for training
        self._load_data(train=True)  # 加载所有环境的数据生成器和成本地图
        if self._cfg.hierarchical:  # 如果是分层训练
            step_counter = 0  # 初始化分层步数计数器
            train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)  # 获取初始阶段的数据加载器
        else:
            train_loader_list, val_loader_list = self._get_dataloader()  # 获取标准数据加载器

        try:
            wandb.watch(self.net)  # 监控模型的梯度和参数
        except:  # noqa: E722
            print("[WARNING] Wandb model watch failed")

        for epoch in range(self._cfg.epochs):  # 主训练循环，遍历所有epoch
            train_loss = 0  # 当前epoch的训练损失
            val_loss = 0  # 当前epoch的验证损失
            for i in range(len(train_loader_list)):  # 遍历每个环境的数据加载器
                train_loss += self._train_epoch(train_loader_list[i], epoch, env_id=i)  # 在第i个环境上训练一个epoch
                val_loss += self._test_epoch(val_loader_list[i], env_id=i, epoch=epoch)  # 在第i个环境上验证一个epoch

            train_loss /= len(train_loader_list)  # 计算平均训练损失
            val_loss /= len(train_loader_list)  # 计算平均验证损失

            try:
                wandb.log( # 记录当前epoch的训练和验证损失
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch,
                    }
                )
            except:  # noqa: E722
                print("[WARNING] Wandb logging failed")

            # if val_loss < best_loss:
            if val_loss < self.best_loss:  # 如果当前验证损失是历史最佳
                print("[INFO] Save model of epoch %d" % (epoch))
                torch.save((self.net.state_dict(), val_loss), self.model_path)  # 保存模型状态和损失
                self.best_loss = val_loss  # 更新最佳损失
                print("[INFO] Current val loss: %.4f" % (self.best_loss))

            if self.scheduler.step(val_loss):  # 更新学习率调度器，并检查是否满足早期停止条件
                print("[INFO] Early Stopping!")
                break

            if self._cfg.hierarchical and (epoch + 1) % self._cfg.hierarchical_step == 0:  # 如果是分层训练，并且达到分层步长
                torch.save(  # 保存当前分层阶段的模型
                    (self.net.state_dict(), self.best_loss),
                    os.path.join(
                        self.model_dir_hierarch,
                        (
                            f"model_ep{epoch}_fov{round(self.fov_ratio, 3)}_"
                            f"front{round(self.front_ratio, 3)}_"
                            f"back{round(self.back_ratio, 3)}.pt"  # 模型文件名包含分层参数
                        ),
                    ),
                )
                step_counter += 1  # 分层步数递增
                train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)  # 获取新的数据加载器（数据采样策略已调整）
                self.hierach_losses[epoch] = self.best_loss  # 记录当前分层阶段的最佳损失

        torch.cuda.empty_cache()  # 清理CUDA缓存

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()  # 清理数据生成器资源

        # empty buffers
        self.data_generators = []  # 清空数据生成器列表
        self.data_traj_cost = []  # 清空轨迹成本数据
        self.data_traj_viz = []  # 清空轨迹可视化数据
        return

    def test(self, step: Optional[int] = None) -> None:
        print("[INFO] Start Training")
        # set random seed for reproducibility
        torch.manual_seed(self._cfg.seed)  # 设置随机种子以确保结果可复现

        # define step
        if step is None and self._cfg.hierarchical:   # 如果是分层训练且没有指定测试步长
            step = int(self._cfg.epochs / self._cfg.hierarchical_step)  # 默认使用最后一个分层阶段的配置

        # load model
        self._load_model(resume=True)  # 加载训练好的最佳模型
        # get dataloader for training
        self._load_data(train=False)  # 加载测试环境的数据生成器和成本地图
        _, test_loader = self._get_dataloader(train=False, step=step)  # 获取测试数据加载器

        self.test_loss = self._test_epoch(
            test_loader[0],  # 使用第一个（也是唯一一个）测试环境的加载器
            env_id=0,  # 测试环境的ID
            is_visual=not os.getenv("EXPERIMENT_DIRECTORY"),  # 如果没有设置EXPERIMENT_DIRECTORY环境变量，则进行可视化
            fov_angle=self.data_generators[0].alpha_fov,  # 传入视野角度用于可视化
            dataset="test",
        )

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()  # 清理数据生成器资源

    def save_config(self) -> None:
        print(f"[INFO] val_loss: {self.best_loss:.2f}, test_loss," f"{self.test_loss:.4f}")
        """ Save config and loss to file"""
        path, _ = os.path.splitext(self.model_path)  # 获取模型路径的基名
        yaml_path = path + ".yaml"  # 构建YAML配置文件的路径
        print(f"[INFO] Save config and loss to {yaml_path} file")

        loss_dict = {"val_loss": self.best_loss, "test_loss": self.test_loss}  # 存储最佳验证损失和测试损失
        save_dict = {"config": vars(self._cfg), "loss": loss_dict}  # 将配置和损失打包成一个字典

        # dump yaml
        with open(yaml_path, "w+") as file:
            yaml.dump(save_dict, file, allow_unicode=True, default_flow_style=False)

        # logging
        with contextlib.suppress(Exception):  # 忽略可能的异常
            wandb.finish()  # 结束Wandb run

        # plot hierarchical losses
        if self._cfg.hierarchical:  # 如果进行了分层训练
            plt.figure(figsize=(10, 10))
            plt.plot(  # 绘制分层损失图
                list(self.hierach_losses.keys()),
                list(self.hierach_losses.values()),
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Hierarchical Losses")
            plt.savefig(os.path.join(self.model_dir_hierarch, "hierarchical_losses.png"))  # 保存图表
            plt.close()  # 关闭图表以释放内存

        return

    """PRIVATE METHODS"""

    # Helper function DATA
    def _load_data(self, train: bool = True) -> None:
        if not isinstance(self._cfg.data_cfg, list):  # 如果data_cfg不是列表
            self._cfg.data_cfg = [self._cfg.data_cfg] * len(self._cfg.env_list)  # 则复制一份data_cfg给每个环境
        assert len(self._cfg.data_cfg) == len(self._cfg.env_list), (  # 检查data_cfg数量是否与环境数量匹配
            "Either single DataCfg or number matching number of environments" "must be provided"
        )

        for idx, env_name in enumerate(self._cfg.env_list):  # 遍历所有环境
            if (train and idx == self._cfg.test_env_id) or (not train and idx != self._cfg.test_env_id):
                continue  # 根据train标志和test_env_id跳过不相关的环境

            data_path = os.path.join(self._cfg.data_dir, env_name)  # 构建当前环境的数据路径

            # get trajectory cost map
            traj_cost = TrajCost(   # 创建轨迹成本计算器实例
                self._cfg.gpu_id,
                log_data=train,
                w_obs=self._cfg.w_obs,  # 障碍物权重
                w_height=self._cfg.w_height,  # 高度权重
                w_goal=self._cfg.w_goal,  # 目标权重
                w_motion=self._cfg.w_motion,  # 运动成本权重
                obstalce_thread=self._cfg.obstacle_thread,  # 障碍物阈值
            )
            traj_cost.SetMap(  # 为轨迹成本计算器设置成本地图
                data_path,
                self._cfg.cost_map_name,
            )

            generator = PlannerDataGenerator(  # 创建数据生成器实例
                cfg=self._cfg.data_cfg[idx],
                root=data_path,
                semantics=self._cfg.sem,  # 是否使用语义信息
                rgb=self._cfg.rgb,  # 是否使用RGB信息
                cost_map=traj_cost.cost_map,  # trajectory cost class  传入成本地图
            )

            traj_viz = TrajViz(  # 创建轨迹可视化实例
                intrinsics=generator.K_depth,  # 深度相机内参
                cam_resolution=self._cfg.img_input_size,  # 相机分辨率
                camera_tilt=self._cfg.camera_tilt,  # 相机倾斜角度
                cost_map=traj_cost.cost_map,  # 成本地图
            )

            self.data_generators.append(generator)  # 将实例添加到列表中
            self.data_traj_cost.append(traj_cost)
            self.data_traj_viz.append(traj_viz)
            print(f"LOADED DATA FOR ENVIRONMENT: {env_name}")

        print("[INFO] LOADED ALL DATA")
        return

    # Helper function TRAINING
    def _init_logging(self) -> None:
        # logging
        os.environ["WANDB_API_KEY"] = self._cfg.wb_api_key  # 设置Wandb API Key
        os.environ["WANDB_MODE"] = "online"  # 设置Wandb模式为在线
        os.makedirs(self._cfg.log_dir, exist_ok=True)  # 创建日志目录（如果不存在）

        try:
            wandb.init(  # 初始化Wandb
                project=self._cfg.wb_project,  # 项目名称
                entity=self._cfg.wb_entity,  # 实体名称
                name=self._cfg.get_model_save(),  # 运行名称
                config=self._cfg.__dict__,  # 配置参数
                dir=self._cfg.log_dir,  # 日志目录
            )
        except:  # noqa: E722
            print("[WARNING: Wandb not available")
        return

    def _load_model(self, resume: bool = False) -> None:
        if self._cfg.sem or self._cfg.rgb:  # 如果使用语义或RGB输入
            if self._cfg.rgb and self._cfg.pre_train_sem:  # 如果使用RGB并且需要预训练语义模型
                assert PRE_TRAIN_POSSIBLE, (  # 检查是否可以使用预训练模型
                    "Pretrained model not available since either detectron2"
                    " not installed or mask2former not found in thrid_party"
                    " folder"
                )
                pre_train_cfg = os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_cfg)  # 预训练配置文件路径
                pre_train_weights = (
                    os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_weights)
                    if self._cfg.pre_train_weights
                    else None
                )  # 预训练权重路径，如果未指定则为None
                m2f_cfg = get_m2f_cfg(pre_train_cfg)  # 获取Mask2Former配置
                self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN  # 获取像素均值
                self.pixel_std = m2f_cfg.MODEL.PIXEL_STD  # 获取像素标准差
            else:
                m2f_cfg = None
                pre_train_weights = None

            self.net = DualAutoEncoder(self._cfg, m2f_cfg=m2f_cfg, weight_path=pre_train_weights)  # 创建双输入编码器-解码器模型实例
        else:  # 如果只使用深度图输入
            self.net = AutoEncoder(self._cfg.in_channel, self._cfg.knodes)  # 创建单输入编码器-解码器模型实例

        assert torch.cuda.is_available(), "Code requires GPU"  # 确保CUDA可用
        print(f"Available GPU list: {list(range(torch.cuda.device_count()))}")
        print(f"Running on GPU: {self._cfg.gpu_id}")
        self.net = self.net.cuda(self._cfg.gpu_id)  # 将模型移动到指定GPU
        print(f"[INFO] MODEL LOADED ({count_parameters(self.net)} parameters)")

        if resume:  # 如果需要从检查点恢复
            model_state_dict, self.best_loss = torch.load(self.model_path)  # 加载模型状态字典和最佳损失
            self.net.load_state_dict(model_state_dict)  # 将状态字典加载到模型中
            print(f"Resume train from {self.model_path} with loss " f"{self.best_loss}")

        return

    def _configure_optimizer(self) -> None:
        if self._cfg.optimizer == "adam":  # 如果使用Adam优化器
            self.optimizer = optim.Adam(  # 创建Adam优化器实例
                self.net.parameters(),
                lr=self._cfg.lr, # 学习率
                weight_decay=self._cfg.w_decay,  # 权重衰减
            )
        elif self._cfg.optimizer == "sgd":  # 如果使用SGD优化器
            self.optimizer = optim.SGD(  # 创建SGD优化器实例
                self.net.parameters(),
                lr=self._cfg.lr,  # 学习率
                momentum=self._cfg.momentum,  # 动量
                weight_decay=self._cfg.w_decay,  # 权重衰减
            )
        else:
            raise KeyError(f"Optimizer {self._cfg.optimizer} not supported")
        self.scheduler = EarlyStopScheduler(  # 创建早期停止调度器实例
            self.optimizer,
            factor=self._cfg.factor,  # 学习率衰减因子
            verbose=True,  # 是否打印日志
            min_lr=self._cfg.min_lr,  # 最小学习率
            patience=self._cfg.patience,  # 容忍度
        )
        print("[INFO] OPTIMIZER AND SCHEDULER CONFIGURED")
        return     

    def _get_dataloader(
        self,
        train: bool = True,
        step: Optional[int] = None,  # 分层训练的当前步数
        allow_augmentation: bool = True,  # 是否允许数据增强
    ) -> None:
        train_loader_list: List[Data.DataLoader] = []
        val_loader_list: List[Data.DataLoader] = []

        if step is not None:    # 如果提供了分层训练步数
            self.fov_ratio = (
                1.0 - (self._cfg.hierarchical_front_step_ratio + self._cfg.hierarchical_back_step_ratio) * step
            )  # 计算视场角比例
            self.front_ratio = self._cfg.hierarchical_front_step_ratio * step  # 计算前方视场比例
            self.back_ratio = self._cfg.hierarchical_back_step_ratio * step  # 计算后方视场比例

        for generator in self.data_generators:  # 遍历所有数据生成器
            # init data classes
            # 每个数据集的生成器对应两个数据集实例：训练集和验证集
            # 第一步，先创建验证数据集实例
            val_data = PlannerData(  # 创建验证数据集实例
                cfg=generator._cfg,
                transform=self.transform,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
            )

            if train:  # 如果是训练模式
                train_data = PlannerData(  # 创建训练数据集实例
                    cfg=generator._cfg,
                    transform=self.transform,
                    semantics=self._cfg.sem,
                    rgb=self._cfg.rgb,
                    pixel_mean=self.pixel_mean,
                    pixel_std=self.pixel_std,
                )
            else:
                train_data = None

            # split data in train and validation with given sample ratios
            # 第二步，使用生成器根据采样比例分割训练和验证数据集
            if train:
                generator.split_samples(  # 分割训练和验证数据集
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,
                    ratio_back_samples=self.back_ratio,  # 后方样本比例
                    ratio_front_samples=self.front_ratio,  # 前方样本比例
                    ratio_fov_samples=self.fov_ratio,  # 视场角样本比例
                    allow_augmentation=allow_augmentation,
                )
            else:  # testing
                generator.split_samples(  # 分割训练和验证数据集
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,  # False
                    ratio_back_samples=self.back_ratio,
                    ratio_front_samples=self.front_ratio,
                    ratio_fov_samples=self.fov_ratio,
                    allow_augmentation=allow_augmentation,
                )

            if self._cfg.load_in_ram:  # 如果配置了将数据加载到内存
                if train:
                    train_data.load_data_in_memory()  # 将训练数据加载到内存
                val_data.load_data_in_memory()  # 将验证数据加载到内存

            # 绑定关系：当你写 Data.DataLoader(dataset=train_data, ...) 时，你实际上是把刚才那个“灌满了特定难度样本”的桶（Dataset）放到了传送带（DataLoader）上。
            if train:
                train_loader = Data.DataLoader(  # 创建训练数据加载器
                    dataset=train_data,
                    batch_size=self._cfg.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=self._cfg.num_workers,
                )
            val_loader = Data.DataLoader(
                dataset=val_data,
                batch_size=self._cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self._cfg.num_workers,
            )

            if train:
                train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)

        return train_loader_list, val_loader_list

    # 具体的训练过程
    def _train_epoch(
        self,
        loader: Data.DataLoader,
        epoch: int,
        env_id: int,
    ) -> float:
        train_loss, batches = 0, len(loader)
        enumerater = tqdm.tqdm(enumerate(loader))  # 进度条显示

        for batch_idx, inputs in enumerater:
            odom = inputs[2].cuda(self._cfg.gpu_id)   # 里程计数据
            goal = inputs[3].cuda(self._cfg.gpu_id)  # 目标位置数据
            self.optimizer.zero_grad()  # 清零梯度

            if self._cfg.sem or self._cfg.rgb:  # 如果使用语义或RGB输入
                depth_image = inputs[0].cuda(self._cfg.gpu_id)  # 深度图像数据
                sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)  # 语义或RGB图像数据
                # 当你写 preds, fear = self.net(image, goal) 时，它最终会去执行你在模型类里定义的 def forward(self, ...): 下面的代码。
                # 在 Python 中，当你像调用函数一样“调用”一个对象（在对象后面加括号 ()）时，Python 会自动去寻找这个对象内部的一个特殊方法：__call__。
                # PyTorch 的所有网络层（包括你自己写的 PlannerNet）都继承自 nn.Module。nn.Module 帮你写好了 __call__ 方法，它的逻辑大概是这样的（伪代码）：
                # # PyTorch 内部的逻辑（简化版）
                # class Module:
                #     def __call__(self, *input, **kwargs):
                #         # 1. 做一些准备工作（比如处理 Hooks，记录操作图等）
                #         # ...
                #         # 2. 【关键】调用你写的 forward 函数
                #         result = self.forward(*input, **kwargs)
                #         # 3. 做一些收尾工作
                #         # ...
                #         return result
                # 既然最终都是运行 forward，为什么不直接写 self.net.forward(image, goal) 呢？
                # 如果不通过 self.net() 这个入口，而是直接走后门去调 forward，你会跳过 PyTorch 帮你做的很多“幕后工作”，也就是上面伪代码里的第 1 步和第 3 步。
                # 这些幕后工作包括：
                # Hooks (挂钩)： 如果你在网络里挂了一些钩子（比如用来提取中间层特征，或者监控梯度），直接调 forward 钩子就不生效了。
                # Just-In-Time (JIT) 编译： 影响模型导出。
                # Amp (自动混合精度)： 可能会影响半精度训练的处理。
                preds, fear = self.net(depth_image, sem_rgb_image, goal)  # 前向传播
            else:  # 仅使用深度图输入
                image = inputs[0].cuda(self._cfg.gpu_id)  # 深度图像数据
                preds, fear = self.net(image, goal)

            # flip y axis for augmented samples  (clone necessary due to
            # inplace operation that otherwise leads to error in backprop)
            preds_flip = torch.clone(preds)  # 克隆预测结果以避免原地操作错误
            preds_flip[inputs[4], :, 1] = preds_flip[inputs[4], :, 1] * -1  # 翻转y轴坐标
            goal_flip = torch.clone(goal)  # 克隆目标位置
            goal_flip[inputs[4], 1] = goal_flip[inputs[4], 1] * -1  # 翻转y轴坐标

            log_step = batch_idx + epoch * batches  # 计算日志记录步数
            loss, _ = self._loss(  # 计算损失
                preds_flip,
                fear,
                self.data_traj_cost[env_id],  # 当前环境的轨迹成本计算器
                odom,
                goal_flip,
                log_step=log_step,
            )
            wandb.log({"train_loss_step": loss}, step=log_step)  # 记录训练损失

            loss.backward()   # 反向传播计算梯度
            self.optimizer.step()  # 更新模型参数
            train_loss += loss.item()  # 累积训练损失
            enumerater.set_description(  # 更新进度条描述
                f"Epoch: {epoch} in Env: "
                f"({env_id+1}/{len(self._cfg.env_list)-1}) "
                f"- train loss:{round(train_loss/(batch_idx+1), 4)} on"
                f" {batch_idx}/{batches}"
            )
        return train_loss / (batch_idx + 1)  # 返回平均训练损失

    def _test_epoch(
        self,
        loader,
        env_id: int,
        epoch: int = 0,
        is_visual=False,  # 是否进行可视化
        fov_angle: float = 90.0,
        dataset: str = "val",  # 数据集类型（验证或测试）
    ) -> float:
        test_loss = 0
        num_batches = len(loader)
        preds_viz = []   # 用于存储可视化的预测结果1
        wp_viz = []    # 用于存储可视化的路径点
        image_viz = []  # 用于存储可视化的图像

        with torch.no_grad():  # 禁用梯度计算
            for batch_idx, inputs in enumerate(loader):
                odom = inputs[2].cuda(self._cfg.gpu_id)
                goal = inputs[3].cuda(self._cfg.gpu_id)

                if self._cfg.sem or self._cfg.rgb:
                    image = inputs[0].cuda(self._cfg.gpu_id)  # depth
                    sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)  # sem
                    preds, fear = self.net(image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self._cfg.gpu_id)
                    preds, fear = self.net(image, goal)

                # flip y axis for augmented samples
                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1  # 翻转y轴坐标
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1  # 翻转y轴坐标

                log_step = epoch * num_batches + batch_idx
                loss, waypoints = self._loss(  # 计算损失
                    preds,
                    fear,
                    self.data_traj_cost[env_id],
                    odom,
                    goal,
                    log_step=log_step,
                    dataset=dataset,
                )

                if dataset == "val":  # 仅在验证集上记录损失
                    wandb.log({f"{dataset}_loss_step": loss}, step=log_step)

                test_loss += loss.item()

                if is_visual and len(preds_viz) * batch_idx < self._cfg.n_visualize:  # 收集可视化数据
                    if batch_idx == 0:  # first batch
                        odom_viz = odom.cpu() # 里程计数据
                        goal_viz = goal.cpu()  # 目标位置
                        fear_viz = fear.cpu()  # 恐惧值
                        augment_viz = inputs[4].cpu()  # 增强标记
                    else:  # concatenate
                        odom_viz = torch.cat((odom_viz, odom.cpu()), dim=0)
                        goal_viz = torch.cat((goal_viz, goal.cpu()), dim=0)
                        fear_viz = torch.cat((fear_viz, fear.cpu()), dim=0)
                        augment_viz = torch.cat((augment_viz, inputs[4].cpu()), dim=0)
                    preds_viz.append(preds.cpu())  # 预测结果
                    wp_viz.append(waypoints.cpu())  # 路径点
                    image_viz.append(image.cpu())  # 输入图像

            if is_visual:
                preds_viz = torch.vstack(preds_viz)  # 将列表中的张量垂直堆叠成一个张量
                wp_viz = torch.vstack(wp_viz)  # 将列表中的张量垂直堆叠成一个张量
                image_viz = torch.vstack(image_viz)  # 将列表中的张量垂直堆叠成一个张量

                # limit again to number of visualizations since before
                # added as multiple of batch size
                preds_viz = preds_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                wp_viz = wp_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                image_viz = image_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                odom_viz = odom_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                goal_viz = goal_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                fear_viz = fear_viz[: self._cfg.n_visualize]  # 截取前n_visualize个样本
                augment_viz = augment_viz[: self._cfg.n_visualize]

                # visual trajectory and images
                self.data_traj_viz[env_id].VizTrajectory(  # 调用轨迹可视化工具
                    preds_viz,
                    wp_viz,
                    odom_viz,
                    goal_viz,
                    fear_viz,
                    fov_angle=fov_angle,
                    augment_viz=augment_viz,
                )
                self.data_traj_viz[env_id].VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)  # 调用图像可视化工具
        return test_loss / (batch_idx + 1)  # 返回平均测试损失

    def _loss(
        self,
        preds: torch.Tensor,  # 预测的路径点张量
        fear: torch.Tensor,  # 恐惧值张量
        traj_cost: TrajCost,  # 轨迹代价计算对象    
        odom: torch.Tensor,     # 里程计张量
        goal: torch.Tensor,     # 目标位置张量
        log_step: int,          # 日志记录步骤
        step: float = 0.1,      # 分层训练的当前步长（用于调整轨迹生成）
        dataset: str = "train",     # 数据集类型（训练或验证）
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 将预测的路径点转换为实际轨迹点，有网络输出量量转化为实际参与评价的轨迹点。跟起止点的速度无关
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)  # 生成轨迹点
        loss = traj_cost.CostofTraj(  # 计算轨迹代价作为损失
            waypoints,
            odom,
            goal,
            fear, # 恐惧值也作为成本的一部分
            log_step,
            ahead_dist=self._cfg.fear_ahead_dist,  # 恐惧值提前距离
            dataset=dataset,
        )

        return loss, waypoints  # 返回总损失和生成的路点


# EoF
