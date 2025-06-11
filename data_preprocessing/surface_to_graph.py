"""
将FreeSurfer输出转换为PyTorch Geometric数据对象
"""

import torch
from torch_geometric.data import Data
from pathlib import Path
import numpy as np
from typing import Union, Optional, Dict
import logging
from utils_surface import (
    read_surface, read_morph_data, read_annot,
    build_adj_matrix, compute_edge_features,
    normalize_features, compute_vertex_normals
)


class SurfaceToGraph:
    def __init__(self,
                 normalize_method: str = 'zscore',
                 include_edge_features: bool = True,
                 include_normals: bool = True,
                 log_level: str = "INFO"):
        """
        初始化转换器

        Args:
            normalize_method: 特征归一化方法
            include_edge_features: 是否包含边特征
            include_normals: 是否包含法向量特征
            log_level: 日志级别
        """
        self.normalize_method = normalize_method
        self.include_edge_features = include_edge_features
        self.include_normals = include_normals

        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _load_surface_data(self,
                           subject_dir: Path,
                           hemi: str) -> Dict:
        """
        加载单个半球的表面数据

        Args:
            subject_dir: 被试目录
            hemi: 半球 ('lh' 或 'rh')

        Returns:
            dict: 加载的数据
        """
        data = {}

        # 加载pial表面
        pial_file = subject_dir / 'surf' / f'{hemi}.pial'
        data['vertices'], data['faces'] = read_surface(pial_file)

        # 加载形态学特征
        for feat in ['curv', 'sulc', 'thickness']:
            feat_file = subject_dir / 'surf' / f'{hemi}.{feat}'
            data[feat] = read_morph_data(feat_file)

        # 加载标注
        annot_file = subject_dir / 'label' / f'{hemi}.aparc.annot'
        data['labels'], _, _ = read_annot(annot_file)

        return data

    def convert_to_pyg(self,
                       subject_dir: Union[str, Path],
                       hemi: str) -> Data:
        """
        转换为PyG Data对象

        Args:
            subject_dir: 被试目录
            hemi: 半球 ('lh' 或 'rh')

        Returns:
            Data: PyG数据对象
        """
        subject_dir = Path(subject_dir)
        self.logger.info(f"处理被试: {subject_dir.name}, 半球: {hemi}")

        # 加载数据
        surface_data = self._load_surface_data(subject_dir, hemi)

        # 构建节点特征
        node_features = []

        # 添加坐标特征（居中化）
        vertices = surface_data['vertices']
        vertices = vertices - vertices.mean(axis=0)
        node_features.append(vertices)

        # 添加形态学特征（归一化）
        for feat in ['curv', 'sulc', 'thickness']:
            norm_feat = normalize_features(surface_data[feat],
                                           method=self.normalize_method)
            node_features.append(norm_feat)

        # 可选：添加法向量
        if self.include_normals:
            normals = compute_vertex_normals(vertices, surface_data['faces'])
            node_features.append(normals)

        # 合并所有节点特征
        node_features = np.hstack([f.reshape(len(vertices), -1)
                                   for f in node_features])

        # 构建边索引和特征
        adj_matrix = build_adj_matrix(surface_data['faces'], len(vertices))
        edges = np.array(adj_matrix.nonzero())

        # 可选：计算边特征
        if self.include_edge_features:
            edge_features = compute_edge_features(
                vertices,
                list(zip(edges[0], edges[1]))
            )
            edge_features = normalize_features(edge_features,
                                               method=self.normalize_method)
        else:
            edge_features = None

        # 创建PyG Data对象
        data = Data(
            x=torch.FloatTensor(node_features),
            edge_index=torch.LongTensor(edges),
            edge_attr=torch.FloatTensor(edge_features) if edge_features is not None else None,
            labels=torch.LongTensor(surface_data['labels'])
        )

        return data

    def process_and_save(self,
                         subject_dir: Union[str, Path],
                         output_dir: Union[str, Path],
                         hemi: str) -> Path:
        """
        处理并保存为.pt文件

        Args:
            subject_dir: 被试目录
            output_dir: 输出目录
            hemi: 半球 ('lh' 或 'rh')

        Returns:
            Path: 保存的文件路径
        """
        subject_dir = Path(subject_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 转换为PyG Data
        data = self.convert_to_pyg(subject_dir, hemi)

        # 保存文件
        output_file = output_dir / f"{subject_dir.name}_{hemi}.pt"
        torch.save(data, output_file)
        self.logger.info(f"保存到: {output_file}")

        return output_file


def main():
    import argparse

    parser = argparse.ArgumentParser(description='FreeSurfer数据转PyG格式')
    parser.add_argument('--subjects_dir', required=True, help='FreeSurfer subjects目录')
    parser.add_argument('--output_dir', required=True, help='输出目录')
    parser.add_argument('--subject_ids', required=True, help='被试ID列表文件')
    parser.add_argument('--normalize', default='zscore', help='归一化方法')
    args = parser.parse_args()

    converter = SurfaceToGraph(normalize_method=args.normalize)

    with open(args.subject_ids) as f:
        subject_ids = [line.strip() for line in f]

    for subject_id in subject_ids:
        subject_dir = Path(args.subjects_dir) / subject_id
        for hemi in ['lh', 'rh']:
            converter.process_and_save(subject_dir, args.output_dir, hemi)


if __name__ == '__main__':
    main()