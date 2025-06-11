"""
FreeSurfer表面数据处理的工具函数
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import nibabel as nib
from scipy.sparse import csr_matrix
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def read_surface(surface_file: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    读取FreeSurfer的surface文件(.pial等)

    Args:
        surface_file: 文件路径

    Returns:
        vertices: [N, 3] 顶点坐标数组
        faces: [M, 3] 三角形面片索引数组
    """
    vertices, faces = nib.freesurfer.read_geometry(str(surface_file))
    return vertices, faces


def read_morph_data(morph_file: Path) -> np.ndarray:
    """
    读取形态学数据文件(.curv, .sulc, .thickness等)

    Args:
        morph_file: 文件路径

    Returns:
        np.ndarray: 形态学数据数组
    """
    return nib.freesurfer.read_morph_data(str(morph_file))


def read_annot(annot_file: Path) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    读取标注文件(.annot)

    Args:
        annot_file: 文件路径

    Returns:
        labels: 每个顶点的标签
        ctab: 颜色表
        names: 区域名称字典
    """
    return nib.freesurfer.read_annot(str(annot_file))


def build_adj_matrix(faces: np.ndarray, num_vertices: int) -> csr_matrix:
    """
    根据面片信息构建邻接矩阵

    Args:
        faces: [M, 3] 三角形面片索引数组
        num_vertices: 顶点总数

    Returns:
        csr_matrix: 稀疏邻接矩阵
    """
    edges = set()
    for face in faces:
        edges.add(tuple(sorted([face[0], face[1]])))
        edges.add(tuple(sorted([face[1], face[2]])))
        edges.add(tuple(sorted([face[2], face[0]])))

    edges = list(edges)
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(rows))

    return csr_matrix((data, (rows, cols)), shape=(num_vertices, num_vertices))


def compute_edge_features(vertices: np.ndarray,
                          edges: List[Tuple[int, int]]) -> np.ndarray:
    """
    计算边特征

    Args:
        vertices: [N, 3] 顶点坐标
        edges: 边列表 [(v1, v2), ...]

    Returns:
        edge_features: [E, D] 边特征矩阵
    """
    # 计算欧氏距离
    edge_features = []
    for v1, v2 in edges:
        dist = np.linalg.norm(vertices[v1] - vertices[v2])
        edge_features.append([dist])
    return np.array(edge_features)


def normalize_features(features: np.ndarray,
                       method: str = 'zscore') -> np.ndarray:
    """
    特征归一化

    Args:
        features: [N, D] 特征矩阵
        method: 归一化方法 ('zscore' 或 'minmax')

    Returns:
        normalized: 归一化后的特征
    """
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"未知的归一化方法: {method}")

    if features.ndim == 1:
        features = features.reshape(-1, 1)
    return scaler.fit_transform(features)


def compute_vertex_normals(vertices: np.ndarray,
                           faces: np.ndarray) -> np.ndarray:
    """
    计算顶点法向量

    Args:
        vertices: [N, 3] 顶点坐标
        faces: [M, 3] 面片索引

    Returns:
        normals: [N, 3] 顶点法向量
    """
    # 计算面片法向量
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    # 归一化面片法向量
    face_normals = face_normals / np.linalg.norm(face_normals, axis=1, keepdims=True)

    # 累加到顶点
    vertex_normals = np.zeros_like(vertices)
    for i, face in enumerate(faces):
        vertex_normals[face] += face_normals[i]

    # 归一化顶点法向量
    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals, axis=1, keepdims=True)

    return vertex_normals