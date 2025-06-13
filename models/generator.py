import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, TopKPooling
from typing import List

class GATConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, heads=4, dropout=0.2, batch_norm=True):
        super().__init__()
        print(f"[GATConvBlock INIT] in_channels={in_channels}, out_channels={out_channels}, heads={heads}")
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
        print(f"[GATConvBlock FORWARD] x.shape={x.shape}")
        x = self.gat(x, edge_index, edge_attr)
        if self.batch_norm:
            x = self.batch_norm(x)
        x = self.activation(x)
        print(f"[GATConvBlock OUT] x.shape={x.shape}")
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ratio=0.5, heads=4):
        super().__init__()
        print(f"[EncoderBlock INIT] in_channels={in_channels}, out_channels={out_channels}, pool_ratio={pool_ratio}")
        self.conv = GATConvBlock(in_channels, out_channels, heads)
        self.pool = TopKPooling(out_channels, ratio=pool_ratio)
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        print(f"[EncoderBlock FORWARD] x.shape={x.shape}")
        x = self.conv(x, edge_index, edge_attr)
        x, edge_index, edge_attr, batch, perm, score = self.pool(
            x, edge_index, edge_attr, batch
        )
        print(f"[EncoderBlock OUT] x.shape={x.shape}")
        return x, edge_index, edge_attr, batch, perm, score

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, heads=4):
        super().__init__()
        print(f"[DecoderBlock INIT] in_channels={in_channels}, skip_channels={skip_channels}, out_channels={out_channels}")
        self.conv = GATConvBlock(in_channels + skip_channels, out_channels, heads)
    def forward(self, x, x_skip, edge_index, edge_attr=None):
        print(f"[DecoderBlock FORWARD] x.shape={x.shape}, x_skip.shape={x_skip.shape}")
        x = torch.cat([x, x_skip], dim=-1)
        print(f"[DecoderBlock CAT] after cat x.shape={x.shape}")
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

        print(f"[Generator INIT] in_channels={in_channels}, hidden_channels={hidden_channels}, pool_ratios={pool_ratios}, heads={heads}")

        # 编码器
        self.encoder_blocks = nn.ModuleList()
        curr_channels = in_channels
        encoder_out_channels = []
        for idx, (h_dim, p_ratio) in enumerate(zip(hidden_channels[:-1], pool_ratios)):
            print(f"  [Generator] Add EncoderBlock {idx}: in={curr_channels}, out={h_dim}, pool={p_ratio}")
            self.encoder_blocks.append(EncoderBlock(curr_channels, h_dim, p_ratio, heads))
            encoder_out_channels.append(h_dim)
            curr_channels = h_dim

        # 瓶颈
        print(f"  [Generator] Add Bottleneck: in={hidden_channels[-2]}, out={hidden_channels[-1]}")
        self.bottleneck = GATConvBlock(hidden_channels[-2], hidden_channels[-1], heads)

        # 解码器
        skip_channels = encoder_out_channels[::-1]                # [128, 64, 32]
        bottleneck_channels = hidden_channels[-1]                 # 256
        decoder_in_channels = [bottleneck_channels] + skip_channels[:-1]  # [256, 128, 64]
        decoder_out_channels = skip_channels                      # [128, 64, 32]

        self.decoder_blocks = nn.ModuleList()
        for i, (in_c, skip_c, out_c) in enumerate(zip(decoder_in_channels, skip_channels, decoder_out_channels)):
            print(f"  [Generator] Add DecoderBlock {i}: in={in_c}, skip={skip_c}, out={out_c}")
            self.decoder_blocks.append(
                DecoderBlock(in_c, skip_c, out_c, heads)
            )

        print(f"  [Generator] Add OutputLayer: in={hidden_channels[0]}, out={in_channels}")
        self.output_layer = GATConv(hidden_channels[0], in_channels, heads=1)

    def encode(self, x, edge_index, edge_attr=None, batch=None):
        encoded_features = []
        curr_x = x
        curr_edge_index = edge_index
        curr_edge_attr = edge_attr
        curr_batch = batch
        for i, encoder in enumerate(self.encoder_blocks):
            print(f"[Generator ENCODE] block {i}, x.shape={curr_x.shape}")
            encoded_features.append((curr_x, curr_edge_index, curr_edge_attr))
            curr_x, curr_edge_index, curr_edge_attr, curr_batch, _, _ = encoder(
                curr_x, curr_edge_index, curr_edge_attr, curr_batch
            )
        print(f"[Generator ENCODE] bottleneck input x.shape={curr_x.shape}")
        bottleneck = self.bottleneck(curr_x, curr_edge_index, curr_edge_attr)
        print(f"[Generator ENCODE] bottleneck out x.shape={bottleneck.shape}")
        return bottleneck, encoded_features

    def decode(self, x, encoded_features):
        curr_x = x
        for i, (decoder, (skip_x, skip_edge_index, skip_edge_attr)) in enumerate(
            zip(self.decoder_blocks, reversed(encoded_features))
        ):
            print(f"[Generator DECODE] block {i}, curr_x.shape={curr_x.shape}, skip_x.shape={skip_x.shape}")
            # 上采样
            if curr_x.shape[0] != skip_x.shape[0]:
                idx = torch.arange(skip_x.shape[0], device=curr_x.device) % curr_x.shape[0]
                x_upsampled = curr_x[idx]
                print(f"[Generator DECODE] Upsample: {curr_x.shape[0]} -> {x_upsampled.shape[0]}")
            else:
                x_upsampled = curr_x
            curr_x = decoder(x_upsampled, skip_x, skip_edge_index, skip_edge_attr)
        return curr_x

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        print(f"[Generator FORWARD] x.shape={x.shape}")
        x, encoded_features = self.encode(x, edge_index, edge_attr, batch)
        x = self.decode(x, encoded_features)
        x = self.output_layer(x, edge_index)
        print(f"[Generator OUT] x.shape={x.shape}")
        return x
