import torch
from torch_geometric.data import Dataset, DataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
import os
from pathlib import Path
import random
import logging
from typing import List, Tuple, Optional, Dict
from torch_geometric.data import Data
from torch_geometric.data.data import DataEdgeAttr  # 添加这行

# 添加这个函数来设置安全的全局变量
def setup_safe_loader():
    """设置安全的数据加载器"""
    torch.serialization.add_safe_globals([
        ('torch_geometric.data.data', 'Data'),
        ('torch_geometric.data.data', 'DataEdgeAttr'),
        # 如果需要，可以添加更多安全的类型
    ])

class BrainGraphDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 seed: int = 42):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.split = split
        random.seed(seed)
        
        # 设置安全加载器
        setup_safe_loader()
        
        self.file_list = sorted([f for f in self.root_dir.glob("*.pt")])
        
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
        try:
            # 首先尝试使用安全模式加载
            data = torch.load(self.file_list[idx], weights_only=True)
        except Exception as e:
            print(f"安全模式加载失败，使用标准模式: {str(e)}")
            # 如果失败，使用标准模式
            data = torch.load(self.file_list[idx])
            
        # 验证数据结构
        assert data.x.size(0) == data.edge_index.max().item() + 1, \
            f"节点数量与边索引不匹配: nodes={data.x.size(0)}, max_idx={data.edge_index.max().item()}"
        
        # 验证边属性
        if hasattr(data, 'edge_attr'):
            assert data.edge_attr.size(0) == data.edge_index.size(1), \
                f"边特征数量与边数量不匹配: attr={data.edge_attr.size(0)}, edges={data.edge_index.size(1)}"
        
        return data

def create_dataloader(dataset: BrainGraphDataset,
                     batch_size: int,
                     shuffle: bool = True,
                     num_workers: int = 4) -> PyGDataLoader:
    """
    创建PyG的DataLoader
    """
    return PyGDataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        follow_batch=['x', 'edge_index']
    )
