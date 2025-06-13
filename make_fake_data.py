import torch
from torch_geometric.data import Data
import numpy as np
import os

def make_fake_graph(num_nodes=1000, num_edges=4000, num_features=6, num_regions=68):
    # 节点特征
    x = torch.randn(num_nodes, num_features)  # 正态分布模拟

    # 边索引
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # 边特征（比如欧氏距离，可以先用随机数代替）
    edge_attr = torch.rand(num_edges, 1)

    # 区域标签（每个节点分到一个区域，区域数自定）
    labels = torch.randint(0, num_regions, (num_nodes,))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, labels=labels)
    return data

def main():
    out_dir = "fake_data"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(10):  # 生成10个样本
        data = make_fake_graph()
        torch.save(data, os.path.join(out_dir, f"fake_subject_{i}.pt"))
    print(f"Saved 10 fake .pt files in {out_dir}")

if __name__ == "__main__":
    main()