"""
数据集加载和处理模块
- 加载.pt格式的PyG Data对象
- 数据集划分
- 批处理
"""

import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader as PyGDataLoader
import os
from pathlib import Path
import random
import logging
from typing import List, Tuple, Optional, Dict
from torch_geometric.data import Data

class BrainGraphDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 42):
        """
        脑图数据集
        Args:
            root_dir: 包含.pt文件的目录
            split: 'train', 'val', 或 'test'
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            seed: 随机种子
        """
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split

        # 设置随机种子
        random.seed(seed)

        # 获取所有.pt文件
        self.file_list = sorted([f for f in self.root_dir.glob("*.pt")])

        # 划分数据集
        n_total = len(self.file_list)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        if split == 'train':
            self.file_list = self.file_list[:n_train]
        elif split == 'val':
            self.file_list = self.file_list[n_train:n_train + n_val]
        else:  # test
            self.file_list = self.file_list[n_train + n_val:]

        logging.info(f"Loaded {len(self.file_list)} samples for {split} set")

    def len(self) -> int:
        return len(self.file_list)

    def get(self, idx: int) -> Data:
        """加载单个PyG Data对象"""
        # 加载数据
        data = torch.load(self.file_list[idx])
        
        # 验证数据结构
        assert data.x.size(0) == data.edge_index.max().item() + 1, \
            f"节点数量与边索引不匹配: nodes={data.x.size(0)}, max_idx={data.edge_index.max().item()}"
            
        if hasattr(data, 'edge_attr'):
            assert data.edge_attr.size(0) == data.edge_index.size(1), \
                f"边特征数量与边数量不匹配: attr={data.edge_attr.size(0)}, edges={data.edge_index.size(1)}"
                
        # 确保x的数值在合理范围内
        if torch.isnan(data.x).any():
            raise ValueError(f"发现NaN值在节点特征中")
            
        if torch.isinf(data.x).any():
            raise ValueError(f"发现Inf值在节点特征中")
            
        return data

def create_dataloader(dataset: BrainGraphDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 0) -> PyGDataLoader:
    """
    创建DataLoader
    Args:
        dataset: BrainGraphDataset实例
        batch_size: 批大小
        shuffle: 是否打乱数据
        num_workers: 数据加载线程数
    Returns:
        PyG DataLoader
    """
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
