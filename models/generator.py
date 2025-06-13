import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling
from typing import List

class GATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, batch_norm=True):
        super().__init__()
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        self.batch_norm = nn.BatchNorm1d(out_channels) if batch_norm else None
        self.activation = nn.LeakyReLU(0.2)
    def forward(self, x, edge_index, edge_attr=None):
        x = self.gat(x, edge_index, edge_attr)
        if self.batch_norm:
            x = self.batch_norm(x)
        return self.activation(x)

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ratio=0.5, heads=4):
        super().__init__()
        self.conv = GATConvBlock(in_channels, out_channels, heads)
        self.pool = TopKPooling(out_channels, ratio=pool_ratio)
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.conv(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score = self.pool(
            x, edge_index, edge_attr, batch
        )
        return x, edge_index, edge_attr, batch, perm, score

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, heads=4):
        super().__init__()
        self.conv = GATConvBlock(in_channels + skip_channels, out_channels, heads)
    def forward(self, x, x_skip, edge_index, edge_attr=None):
        x = torch.cat([x, x_skip], dim=-1)
        x = self.conv(x, edge_index, edge_attr)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_channels: List[int] = [32, 64, 128, 256],
                 pool_ratios: List[float] = [0.8, 0.6, 0.4],
                 heads: int = 4):
        super().__init__()
        assert len(pool_ratios) == len(hidden_channels) - 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        # 编码器
        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        encoder_out_channels = []
        for h_dim, p_ratio in zip(hidden_channels[:-1], pool_ratios):
            self.encoder_blocks.append(EncoderBlock(curr_channels, h_dim, p_ratio, heads))
            encoder_out_channels.append(h_dim)
            curr_channels = h_dim

        # 瓶颈
        self.bottleneck = GATConvBlock(hidden_channels[-2], hidden_channels[-1], heads)

        # 解码器严格对齐
        skip_channels = encoder_out_channels[::-1]                # [128, 64, 32]
        bottleneck_channels = hidden_channels[-1]                 # 256
        decoder_in_channels = [bottleneck_channels] + skip_channels[:-1]  # [256, 128, 64]
        decoder_out_channels = skip_channels                      # [128, 64, 32]

        self.decoder_blocks = nn.ModuleList()
        for in_c, skip_c, out_c in zip(decoder_in_channels, skip_channels, decoder_out_channels):
            self.decoder_blocks.append(
                DecoderBlock(in_c, skip_c, out_c, heads)
            )

        self.output_layer = GATConv(hidden_channels[0], in_channels, heads=1)

    def encode(self, x, edge_index, edge_attr=None, batch=None):
        encoded_features = []
        curr_x = x
        curr_edge_index = edge_index
        curr_edge_attr = edge_attr
        curr_batch = batch
        for encoder in self.encoder_blocks:
            encoded_features.append((curr_x, curr_edge_index, curr_edge_attr))
            curr_x, curr_edge_index, curr_edge_attr, curr_batch, _, _ = encoder(
                curr_x, curr_edge_index, curr_edge_attr, curr_batch
            )
        bottleneck = self.bottleneck(curr_x, curr_edge_index, curr_edge_attr)
        return bottleneck, encoded_features

    def decode(self, x, encoded_features):
        curr_x = x
        for decoder, (skip_x, skip_edge_index, skip_edge_attr) in zip(
            self.decoder_blocks, reversed(encoded_features)
        ):
            # 上采样
            if curr_x.shape[0] != skip_x.shape[0]:
                idx = torch.arange(skip_x.shape[0], device=curr_x.device) % curr_x.shape[0]
                x_upsampled = curr_x[idx]
            else:
                x_upsampled = curr_x
            curr_x = decoder(x_upsampled, skip_x, skip_edge_index, skip_edge_attr)
        return curr_x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x, encoded_features = self.encode(x, edge_index, edge_attr, batch)
        x = self.decode(x, encoded_features)
        x = self.output_layer(x, edge_index)
        return x
