"""
生成器模型
- 图自编码器结构
- 使用GAT层进行特征提取
- 使用TopK池化进行下采样
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from torch_geometric.nn import GATConv, TopKPooling
import torch.nn.functional as F

class GATConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4):
        """GAT卷积块"""
        super().__init__()
        print(f"[GATConvBlock INIT] in_channels={in_channels}, out_channels={out_channels}, heads={heads}")
        # 确保输出维度正确
        self.gat = GATConv(
            in_channels, 
            out_channels // heads,  # 输出通道数除以头数
            heads=heads, 
            concat=True,  # 使用concat模式
            dropout=0.0
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        print(f"[GATConvBlock FORWARD] x.shape={x.shape}")
        x = self.gat(x, edge_index, edge_attr)
        x = F.elu(x)
        print(f"[GATConvBlock OUT] x.shape={x.shape}")
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, pool_ratio: float = 0.5, heads: int = 4):
        """编码器块"""
        super().__init__()
        print(f"[EncoderBlock INIT] in_channels={in_channels}, out_channels={out_channels}, pool_ratio={pool_ratio}")
        self.conv = GATConvBlock(in_channels, out_channels, heads)
        self.pool = TopKPooling(out_channels, ratio=pool_ratio)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        """前向传播"""
        print(f"[EncoderBlock FORWARD] x.shape={x.shape}")
        x = self.conv(x, edge_index, edge_attr)
        # 确保batch信息正确
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x, edge_index, edge_attr, batch, perm, score = self.pool(
            x, edge_index, edge_attr, batch
        )
        print(f"[EncoderBlock OUT] x.shape={x.shape}")
        return x, edge_index, edge_attr, batch, perm, score

class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, heads: int = 4):
        """解码器块"""
        super().__init__()
        print(f"[DecoderBlock INIT] in_channels={in_channels}, skip_channels={skip_channels}, out_channels={out_channels}")
        self.conv = GATConvBlock(in_channels + skip_channels, out_channels, heads)
        
    def forward(self, x, x_skip, edge_index, edge_attr=None):
        """前向传播"""
        print(f"[DecoderBlock FORWARD] x.shape={x.shape}, x_skip.shape={x_skip.shape}")
        x = torch.cat([x, x_skip], dim=1)
        print(f"[DecoderBlock CAT] after cat x.shape={x.shape}")
        return self.conv(x, edge_index, edge_attr)

class Generator(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 pool_ratios: List[float] = [0.8, 0.6, 0.4],
                 heads: int = 4):
        """
        图生成器
        Args:
            in_channels: 输入特征维度
            hidden_channels: 隐藏层维度列表
            pool_ratios: 池化比例列表
            heads: GAT注意力头数
        """
        super().__init__()
        assert len(pool_ratios) == len(hidden_channels) - 1
        
        print(f"[Generator INIT] in_channels={in_channels}, hidden_channels={hidden_channels}, pool_ratios={pool_ratios}, heads={heads}")
        
        # 编码器
        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        encoder_out_channels = []
        
        for idx, (h_dim, p_ratio) in enumerate(zip(hidden_channels[:-1], pool_ratios)):
            print(f"  [Generator] Add EncoderBlock {idx}: in={curr_channels}, out={h_dim}, pool={p_ratio}")
            self.encoder_blocks.append(
                EncoderBlock(curr_channels, h_dim, p_ratio, heads)
            )
            encoder_out_channels.append(h_dim)
            curr_channels = h_dim
            
        # 瓶颈
        print(f"  [Generator] Add Bottleneck: in={hidden_channels[-2]}, out={hidden_channels[-1]}")
        self.bottleneck = GATConvBlock(hidden_channels[-2], hidden_channels[-1], heads)
        
        # 解码器
        skip_channels = encoder_out_channels[::-1]  # 反转编码器输出通道
        bottleneck_channels = hidden_channels[-1]
        decoder_in_channels = [bottleneck_channels] + skip_channels[:-1]
        decoder_out_channels = skip_channels
        
        self.decoder_blocks = nn.ModuleList()
        for i, (in_c, skip_c, out_c) in enumerate(zip(decoder_in_channels, skip_channels, decoder_out_channels)):
            print(f"  [Generator] Add DecoderBlock {i}: in={in_c}, skip={skip_c}, out={out_c}")
            self.decoder_blocks.append(
                DecoderBlock(in_c, skip_c, out_c, heads)
            )
            
        # 输出层
        print(f"  [Generator] Add OutputLayer: in={hidden_channels[0]}, out={in_channels}")
        self.output_layer = GATConv(hidden_channels[0], in_channels, heads=1, concat=False)
        
    def encode(self, x, edge_index, edge_attr=None, batch=None):
        """编码过程"""
        print(f"[Generator ENCODE] block 0, x.shape={x.shape}")
        
        # 保存中间特征
        encoded_features = []
        curr_x = x
        curr_edge_index = edge_index
        curr_edge_attr = edge_attr
        curr_batch = batch if batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        for idx, encoder in enumerate(self.encoder_blocks):
            # 保存当前特征用于跳跃连接
            encoded_features.append((curr_x, curr_edge_index, curr_edge_attr, curr_batch))
            # 编码和池化
            curr_x, curr_edge_index, curr_edge_attr, curr_batch, _, _ = encoder(
                curr_x, curr_edge_index, curr_edge_attr, curr_batch
            )
            print(f"[Generator ENCODE] block {idx+1}, x.shape={curr_x.shape}")
            
        # 瓶颈层
        print(f"[Generator ENCODE] bottleneck input x.shape={curr_x.shape}")
        bottleneck = self.bottleneck(curr_x, curr_edge_index, curr_edge_attr)
        print(f"[Generator ENCODE] bottleneck out x.shape={bottleneck.shape}")
        
        return bottleneck, encoded_features
        
    def decode(self, x, encoded_features):
        """解码过程"""
        curr_x = x
        
        for idx, (decoder, (skip_x, skip_edge_index, skip_edge_attr, _)) in enumerate(
            zip(self.decoder_blocks, reversed(encoded_features))
        ):
            print(f"[Generator DECODE] block {idx}, curr_x.shape={curr_x.shape}, skip_x.shape={skip_x.shape}")
            
            # 处理节点数不匹配
            if curr_x.size(0) < skip_x.size(0):
                print(f"[Generator DECODE] Upsample: {curr_x.size(0)} -> {skip_x.size(0)}")
                # 使用最近邻插值进行上采样
                ratio = skip_x.size(0) // curr_x.size(0)
                curr_x = curr_x.unsqueeze(0)  # [1, N, C]
                curr_x = curr_x.repeat(ratio, 1, 1)  # [ratio, N, C]
                curr_x = curr_x.view(-1, curr_x.size(-1))  # [ratio*N, C]
                
                # 处理余数
                if curr_x.size(0) < skip_x.size(0):
                    remaining = skip_x.size(0) - curr_x.size(0)
                    curr_x = torch.cat([curr_x, curr_x[:remaining]], dim=0)
            
            # 解码
            curr_x = decoder(curr_x, skip_x, skip_edge_index, skip_edge_attr)
        
        return curr_x
        
    def forward(self, x, edge_index, edge_attr=None):
        """前向传播"""
        print(f"[Generator FORWARD] x.shape={x.shape}")
        x, encoded_features = self.encode(x, edge_index, edge_attr)
        x = self.decode(x, encoded_features)
        x = self.output_layer(x, edge_index)
        print(f"[Generator OUT] x.shape={x.shape}")
        return x
