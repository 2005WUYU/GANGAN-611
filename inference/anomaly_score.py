"""
异常分数计算模块
- 节点级分数
- 区域级分数
- 全脑级分数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path
import numpy as np
from training.losses import SpectralLoss


class AnomalyScorer:
    def __init__(self,
                 spectral_k: int = 10,
                 device: torch.device = torch.device('cuda')):
        """
        异常分数计算器
        Args:
            spectral_k: 谱距离使用的特征值数量
            device: 计算设备
        """
        self.spectral_k = spectral_k
        self.device = device
        self.spectral_loss = SpectralLoss(k=spectral_k)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def compute_l1_score(self,
                         x_input: torch.Tensor,
                         x_recon: torch.Tensor) -> torch.Tensor:
        """
        计算L1重建分数
        Args:
            x_input: 输入特征 [N, F]
            x_recon: 重建特征 [N, F]
        Returns:
            节点级L1分数 [N]
        """
        return torch.norm(x_input - x_recon, p=1, dim=1)

    def compute_spectral_score(self,
                               edge_index: torch.Tensor,
                               x_input: torch.Tensor,
                               x_recon: torch.Tensor,
                               edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算谱距离分数
        Args:
            edge_index: 边索引 [2, E]
            x_input: 输入特征 [N, F]
            x_recon: 重建特征 [N, F]
            edge_attr: 边特征（可选）
        Returns:
            节点级谱距离分数 [N]
        """
        # 获取拉普拉斯矩阵
        L_input = get_laplacian(edge_index, edge_attr,
                                normalization='sym',
                                num_nodes=x_input.size(0))
        L_recon = get_laplacian(edge_index, edge_attr,
                                normalization='sym',
                                num_nodes=x_recon.size(0))

        # 计算每个节点的局部谱距离
        scores = []
        for i in range(x_input.size(0)):
            # 提取局部子图
            mask = torch.zeros(x_input.size(0), dtype=torch.bool)
            mask[i] = True
            mask[edge_index[1][edge_index[0] == i]] = True  # 添加邻居

            # 计算局部谱距离
            local_score = self.spectral_loss(
                edge_index[:, mask],
                edge_index[:, mask],
                mask.sum()
            )
            scores.append(local_score)

        return torch.tensor(scores, device=self.device)

    def compute_adversarial_score(self,
                                  discriminator: nn.Module,
                                  x_recon: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算对抗性分数
        Args:
            discriminator: 判别器模型
            x_recon: 重建特征 [N, F]
            edge_index: 边索引 [2, E]
            labels: 区域标签（可选）[N]
        Returns:
            节点级对抗性分数 [N]
        """
        with torch.no_grad():
            scores = discriminator(x_recon, edge_index, labels)
            # 将patch级分数映射回节点级
            if labels is not None:
                # 使用区域标签进行映射
                node_scores = torch.zeros(x_recon.size(0), device=self.device)
                for i, score in enumerate(scores):
                    node_scores[labels == i] = score
            else:
                # 简单复制patch分数给相应节点
                node_scores = scores.repeat_interleave(
                    x_recon.size(0) // scores.size(0)
                )
        return -node_scores  # 取负值，使得分数越高表示越异常

    def compute_all_scores(self,
                           x_input: torch.Tensor,
                           x_recon: torch.Tensor,
                           edge_index: torch.Tensor,
                           discriminator: nn.Module,
                           edge_attr: Optional[torch.Tensor] = None,
                           labels: Optional[torch.Tensor] = None,
                           weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """
        计算所有类型的分数
        Args:
            x_input: 输入特征
            x_recon: 重建特征
            edge_index: 边索引
            discriminator: 判别器模型
            edge_attr: 边特征（可选）
            labels: 区域标签（可选）
            weights: 各类分数的权重
        Returns:
            各类分数字典
        """
        if weights is None:
            weights = {'l1': 1.0, 'spectral': 1.0, 'adversarial': 1.0}

        # 计算各类分数
        scores = {
            'l1': self.compute_l1_score(x_input, x_recon),
            'spectral': self.compute_spectral_score(
                edge_index, x_input, x_recon, edge_attr
            ),
            'adversarial': self.compute_adversarial_score(
                discriminator, x_recon, edge_index, labels
            )
        }

        # 归一化每类分数
        for k in scores:
            scores[k] = (scores[k] - scores[k].mean()) / scores[k].std()

        # 计算加权总分
        scores['total'] = sum(
            weights[k] * scores[k] for k in ['l1', 'spectral', 'adversarial']
        )

        return scores

    def compute_region_scores(self,
                              node_scores: Dict[str, torch.Tensor],
                              labels: torch.Tensor,
                              method: str = 'mean') -> Dict[str, Dict[int, float]]:
        """
        计算区域级分数
        Args:
            node_scores: 节点级分数字典
            labels: 区域标签
            method: 聚合方法
        Returns:
            区域级分数字典
        """
        region_scores = {}
        for score_type, scores in node_scores.items():
            region_scores[score_type] = {}
            unique_labels = torch.unique(labels)

            for label in unique_labels:
                mask = (labels == label)
                region_nodes = scores[mask]

                if method == 'mean':
                    score = region_nodes.mean().item()
                elif method == 'max':
                    score = region_nodes.max().item()
                else:
                    score = region_nodes.median().item()

                region_scores[score_type][label.item()] = score

        return region_scores

    def compute_global_scores(self,
                              node_scores: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
        """
        计算全脑级分数
        Args:
            node_scores: 节点级分数字典
        Returns:
            全脑级统计量字典
        """
        global_scores = {}
        for score_type, scores in node_scores.items():
            global_scores[score_type] = {
                'mean': scores.mean().item(),
                'std': scores.std().item(),
                'max': scores.max().item(),
                'median': scores.median().item(),
                'q75': torch.quantile(scores, 0.75).item(),
                'q90': torch.quantile(scores, 0.90).item()
            }
        return global_scores