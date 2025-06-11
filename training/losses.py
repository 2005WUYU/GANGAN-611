"""
损失函数实现
- WGAN-GP对抗损失
- L1重建损失
- 谱距离损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_dense_adj
import scipy.sparse.linalg as spla
from typing import Tuple, Optional


class GANLoss:
    def __init__(self,
                 lambda_gp: float = 10.0,
                 device: torch.device = torch.device('cuda')):
        """
        WGAN-GP损失
        Args:
            lambda_gp: 梯度惩罚系数
            device: 计算设备
        """
        self.lambda_gp = lambda_gp
        self.device = device

    def discriminator_loss(self,
                           real_scores: torch.Tensor,
                           fake_scores: torch.Tensor,
                           real_data: torch.Tensor,
                           fake_data: torch.Tensor,
                           discriminator: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        判别器损失
        Returns:
            loss: 总损失
            gp: 梯度惩罚项
        """
        # Wasserstein损失
        loss = fake_scores.mean() - real_scores.mean()

        # 计算梯度惩罚
        alpha = torch.rand(real_data.size(0), 1, device=self.device)
        alpha = alpha.expand(real_data.size())

        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)

        interpolated_scores = discriminator(interpolated)
        gradients = torch.autograd.grad(
            outputs=interpolated_scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(interpolated_scores),
            create_graph=True,
            retain_graph=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        loss = loss + self.lambda_gp * gradient_penalty

        return loss, gradient_penalty

    def generator_loss(self, fake_scores: torch.Tensor) -> torch.Tensor:
        """生成器损失"""
        return -fake_scores.mean()


class SpectralLoss:
    def __init__(self, k: int = 10):
        """
        谱距离损失
        Args:
            k: 使用前k个特征值
        """
        self.k = k

    def __call__(self,
                 edge_index_real: torch.Tensor,
                 edge_index_fake: torch.Tensor,
                 num_nodes: int,
                 edge_weight_real: Optional[torch.Tensor] = None,
                 edge_weight_fake: Optional[torch.Tensor] = None) -> torch.Tensor:
        """计算谱距离"""
        # 获取拉普拉斯矩阵
        L_real = get_laplacian(edge_index_real, edge_weight_real,
                               normalization='sym', num_nodes=num_nodes)
        L_fake = get_laplacian(edge_index_fake, edge_weight_fake,
                               normalization='sym', num_nodes=num_nodes)

        # 转换为稠密矩阵
        L_real = to_dense_adj(L_real[0], edge_attr=L_real[1])[0]
        L_fake = to_dense_adj(L_fake[0], edge_attr=L_fake[1])[0]

        # 计算特征值
        eig_real = torch.linalg.eigvalsh(L_real)[:self.k]
        eig_fake = torch.linalg.eigvalsh(L_fake)[:self.k]

        return F.mse_loss(eig_real, eig_fake)


def l1_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """L1重建损失"""
    return F.l1_loss(input, target)