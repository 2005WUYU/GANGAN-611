"""
推理辅助工具
- 批处理工具
- 统计函数
- 数据结构转换
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import nibabel as nib
from torch_geometric.data import Data, Batch
import logging


class InferenceHelper:
    def __init__(self, device: torch.device = torch.device('cuda')):
        self.device = device
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def load_graph_batch(file_list: List[Path]) -> List[Data]:
        """
        加载多个PyG Data对象
        Args:
            file_list: .pt文件路径列表
        Returns:
            加载的Data对象列表
        """
        graphs = []
        for file_path in file_list:
            if not file_path.exists():
                raise FileNotFoundError(f"找不到文件: {file_path}")
            data = torch.load(file_path)
            graphs.append(data)
        return graphs

    @staticmethod
    def create_batch(graphs: List[Data]) -> Batch:
        """
        创建批处理
        Args:
            graphs: Data对象列表
        Returns:
            PyG Batch对象
        """
        return Batch.from_data_list(graphs)

    @staticmethod
    def batch_to_list(batch: Batch) -> List[Data]:
        """
        将批处理拆分回列表
        Args:
            batch: PyG Batch对象
        Returns:
            Data对象列表
        """
        return batch.to_data_list()

    def aggregate_node_scores(self,
                              scores: torch.Tensor,
                              labels: torch.Tensor,
                              method: str = 'mean') -> Dict[int, float]:
        """
        按区域聚合节点分数
        Args:
            scores: 节点分数 [N]
            labels: 节点区域标签 [N]
            method: 聚合方法 ('mean', 'max', 'median')
        Returns:
            区域分数字典 {region_id: score}
        """
        unique_labels = torch.unique(labels)
        region_scores = {}

        for label in unique_labels:
            mask = (labels == label)
            region_nodes = scores[mask]

            if method == 'mean':
                score = region_nodes.mean().item()
            elif method == 'max':
                score = region_nodes.max().item()
            elif method == 'median':
                score = region_nodes.median().item()
            else:
                raise ValueError(f"未知的聚合方法: {method}")

            region_scores[label.item()] = score

        return region_scores

    @staticmethod
    def compute_global_stats(scores: torch.Tensor) -> Dict[str, float]:
        """
        计算全局统计量
        Args:
            scores: 节点分数 [N]
        Returns:
            统计指标字典
        """
        return {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'max': scores.max().item(),
            'median': scores.median().item(),
            'q75': scores.quantile(0.75).item(),
            'q90': scores.quantile(0.90).item()
        }

    @staticmethod
    def create_summary_df(node_scores: torch.Tensor,
                          region_scores: Dict[int, float],
                          global_stats: Dict[str, float],
                          region_names: Optional[Dict[int, str]] = None) -> pd.DataFrame:
        """
        创建汇总DataFrame
        Args:
            node_scores: 节点分数 [N]
            region_scores: 区域分数字典
            global_stats: 全局统计量
            region_names: 区域名称字典（可选）
        Returns:
            汇总DataFrame
        """
        # 创建区域级数据
        region_data = []
        for region_id, score in region_scores.items():
            row = {
                'level': 'region',
                'id': region_id,
                'name': region_names.get(region_id, f"Region_{region_id}") if region_names else f"Region_{region_id}",
                'score': score
            }
            region_data.append(row)

        # 创建全局级数据
        global_data = [{
            'level': 'global',
            'id': 'global',
            'name': stat_name,
            'score': stat_value
        } for stat_name, stat_value in global_stats.items()]

        # 合并数据
        df = pd.DataFrame(region_data + global_data)
        return df

    def save_results(self,
                     output_dir: Path,
                     subject_id: str,
                     node_scores: torch.Tensor,
                     region_scores: Dict[int, float],
                     global_stats: Dict[str, float],
                     suffix: str = '') -> None:
        """
        保存结果
        Args:
            output_dir: 输出目录
            subject_id: 被试ID
            node_scores: 节点分数
            region_scores: 区域分数
            global_stats: 全局统计量
            suffix: 文件名后缀
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存节点分数
        torch.save(
            node_scores,
            output_dir / f"{subject_id}_node_scores{suffix}.pt"
        )

        # 保存区域分数和全局统计量
        results = {
            'region_scores': region_scores,
            'global_stats': global_stats
        }
        torch.save(
            results,
            output_dir / f"{subject_id}_summary{suffix}.pt"
        )

        # 保存CSV格式的汇总
        df = self.create_summary_df(node_scores, region_scores, global_stats)
        df.to_csv(
            output_dir / f"{subject_id}_summary{suffix}.csv",
            index=False
        )