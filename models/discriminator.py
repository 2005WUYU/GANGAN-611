"""
PatchGAN判别器实现
- 基于GCN的特征提取
- 谱归一化保证稳定性
- 支持局部/区域判别策略
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn.utils import spectral_norm
import numpy as np
from typing import List, Optional, Tuple


class SpectralGCNConv(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_spectral_norm: bool = True):
        """
        谱归一化GCN卷积层
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            use_spectral_norm: 是否使用谱归一化
        """
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)

        # 应用谱归一化
        if use_spectral_norm:
            self.conv.lin = spectral_norm(self.conv.lin)

    def forward(self, x, edge_index, edge_weight=None):
        return self.conv(x, edge_index, edge_weight)


class GCNBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 use_spectral_norm: bool = True,
                 dropout: float = 0.2):
        """
        GCN块：卷积+归一化+激活
        Args:
            in_channels: 输入特征维度
            out_channels: 输出特征维度
            use_spectral_norm: 是否使用谱归一化
            dropout: Dropout率
        """
        super().__init__()
        self.conv = SpectralGCNConv(in_channels, out_channels, use_spectral_norm)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_weight=None):
        x = self.conv(x, edge_index, edge_weight)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x)


class PatchDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_channels: List[int] = [64, 128, 256, 512],
                 patch_method: str = 'random',
                 num_patches: int = 64,
                 use_spectral_norm: bool = True):
        """
        PatchGAN判别器
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度列表
            patch_method: 分patch方式 ('random' 或 'region')
            num_patches: random方式时的patch数量
            use_spectral_norm: 是否使用谱归一化
        """
        super().__init__()
        self.patch_method = patch_method
        self.num_patches = num_patches

        # GCN层
        self.conv_blocks = nn.ModuleList()
        curr_channels = in_channels

        for hidden_dim in hidden_channels:
            self.conv_blocks.append(
                GCNBlock(curr_channels, hidden_dim, use_spectral_norm)
            )
            curr_channels = hidden_dim

        # 输出层
        self.output_layer = SpectralGCNConv(
            hidden_channels[-1], 1, use_spectral_norm
        )

    def get_random_patches(self,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机采样子图方式"""
        patch_size = num_nodes // self.num_patches

        patches_x = []
        patches_edge_index = []

        for _ in range(self.num_patches):
            # 随机选择中心节点
            center = np.random.randint(0, num_nodes)

            # 获取邻居节点（k跳邻域）
            neighbor_indices = self._get_k_hop_neighbors(
                center, edge_index, k=2, max_nodes=patch_size
            )

            # 提取子图
            patch_x = x[neighbor_indices]
            patch_edge_index = self._subgraph(edge_index, neighbor_indices)

            patches_x.append(patch_x)
            patches_edge_index.append(patch_edge_index)

        return patches_x, patches_edge_index

    def get_region_patches(self,
                           x: torch.Tensor,
                           edge_index: torch.Tensor,
                           region_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """基于解剖区域的分patch方式"""
        unique_regions = torch.unique(region_labels)
        patches_x = []
        patches_edge_index = []

        for region in unique_regions:
            # 获取该区域的节点
            region_mask = (region_labels == region)
            region_indices = torch.where(region_mask)[0]

            # 提取子图
            patch_x = x[region_indices]
            patch_edge_index = self._subgraph(edge_index, region_indices)

            patches_x.append(patch_x)
            patches_edge_index.append(patch_edge_index)

        return patches_x, patches_edge_index

    @staticmethod
    def _get_k_hop_neighbors(center: int,
                             edge_index: torch.Tensor,
                             k: int,
                             max_nodes: int) -> torch.Tensor:
        """获取k跳邻居"""
        neighbors = {center}
        frontier = {center}

        for _ in range(k):
            new_frontier = set()
            for node in frontier:
                # 获取直接邻居
                mask = (edge_index[0] == node)
                new_neighbors = edge_index[1][mask].tolist()
                new_frontier.update(new_neighbors)

            frontier = new_frontier - neighbors
            neighbors.update(frontier)

            if len(neighbors) >= max_nodes:
                break

        return torch.tensor(list(neighbors)[:max_nodes])

    @staticmethod
    def _subgraph(edge_index: torch.Tensor,
                  node_idx: torch.Tensor) -> torch.Tensor:
        """提取子图的边索引"""
        node_idx = node_idx.tolist()
        mask = torch.zeros(edge_index.max().item() + 1, dtype=torch.bool)
        mask[node_idx] = 1
        edge_mask = mask[edge_index[0]] & mask[edge_index[1]]

        edge_index = edge_index[:, edge_mask]

        # 重新映射节点索引
        node_idx = torch.tensor(node_idx)
        idx = torch.zeros(edge_index.max().item() + 1, dtype=torch.long)
        idx[node_idx] = torch.arange(node_idx.size(0))
        edge_index = idx[edge_index]

        return edge_index

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                region_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            region_labels: 区域标签 [N] (region方式时需要)
        Returns:
            patch级别的真实性得分 [num_patches, 1]
        """
        # 获取patches
        if self.patch_method == 'random':
            patches_x, patches_edge_index = self.get_random_patches(
                x, edge_index, x.size(0)
            )
        else:  # region
            assert region_labels is not None, "Region labels required for region-based patches"
            patches_x, patches_edge_index = self.get_region_patches(
                x, edge_index, region_labels
            )

        # 处理每个patch
        patch_scores = []
        for patch_x, patch_edge_index in zip(patches_x, patches_edge_index):
            # 特征提取
            curr_x = patch_x
            for conv_block in self.conv_blocks:
                curr_x = conv_block(curr_x, patch_edge_index)

            # 输出得分（取平均）
            score = self.output_layer(curr_x, patch_edge_index)
            patch_scores.append(score.mean())

        return torch.stack(patch_scores)