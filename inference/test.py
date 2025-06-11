"""
推理主脚本
- 模型加载
- 批量推理
- 异常分数计算
"""

import torch
from pathlib import Path
import logging
from typing import Dict, List, Optional
import yaml
import argparse
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import PatchDiscriminator
from inference.utils_infer import InferenceHelper
from inference.anomaly_score import AnomalyScorer


class Inferencer:
    def __init__(self,
                 config: Dict,
                 checkpoint_path: Path,
                 device: torch.device = torch.device('cuda')):
        """
        推理器
        Args:
            config: 配置字典
            checkpoint_path: 模型检查点路径
            device: 计算设备
        """
        self.config = config
        self.device = device

        # 初始化日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 初始化模型
        self._init_models()

        # 加载检查点
        self._load_checkpoint(checkpoint_path)

        # 初始化辅助工具
        self.helper = InferenceHelper(device)
        self.scorer = AnomalyScorer(
            spectral_k=config['inference']['spectral_k'],
            device=device
        )

    def _init_models(self):
        """初始化模型"""
        model_config = self.config['model']

        self.generator = Generator(
            in_channels=model_config['in_channels'],
            hidden_channels=model_config['generator_channels'],
            heads=model_config['gat_heads']
        ).to(self.device)

        self.discriminator = PatchDiscriminator(
            in_channels=model_config['in_channels'],
            hidden_channels=model_config['discriminator_channels'],
            patch_method=model_config['patch_method']
        ).to(self.device)

    def _load_checkpoint(self, checkpoint_path: Path):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

        self.generator.eval()
        self.discriminator.eval()

    def process_single(self,
                       data_path: Path,
                       output_dir: Path,
                       save_recon: bool = True) -> Dict:
        """
        处理单个样本
        Args:
            data_path: 输入.pt文件路径
            output_dir: 输出目录
            save_recon: 是否保存重建结果
        Returns:
            处理结果字典
        """
        # 加载数据
        data = torch.load(data_path)
        data = data.to(self.device)

        # 生成重建
        with torch.no_grad():
            x_recon = self.generator(
                data.x, data.edge_index, data.edge_attr
            )

        # 计算异常分数
        node_scores = self.scorer.compute_all_scores(
            data.x, x_recon, data.edge_index,
            self.discriminator,
            data.edge_attr,
            data.labels,
            weights=self.config['inference']['score_weights']
        )

        # 计算区域级分数
        region_scores = self.scorer.compute_region_scores(
            node_scores,
            data.labels,
            method=self.config['inference']['region_method']
        )

        # 计算全脑级分数
        global_scores = self.scorer.compute_global_scores(node_scores)

        # 保存结果
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        subject_id = data_path.stem
        self.helper.save_results(
            output_dir,
            subject_id,
            node_scores,
            region_scores,
            global_scores
        )

        # 可选：保存重建结果
        if save_recon:
            recon_data = data.clone()
            recon_data.x = x_recon
            torch.save(
                recon_data,
                output_dir / f"{subject_id}_recon.pt"
            )

        return {
            'node_scores': node_scores,
            'region_scores': region_scores,
            'global_scores': global_scores
        }

    def process_batch(self,
                      input_dir: Path,
                      output_dir: Path,
                      pattern: str = "*.pt",
                      save_recon: bool = True) -> None:
        """
        批量处理
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            pattern: 文件匹配模式
            save_recon: 是否保存重建结果
        """
        input_dir = Path(input_dir)
        file_list = sorted(input_dir.glob(pattern))

        self.logger.info(f"Found {len(file_list)} files to process")

        for file_path in tqdm(file_list):
            try:
                self.process_single(file_path, output_dir, save_recon)
            except Exception as e:
                self.logger.error(f"Error processing {file_path}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='GAT-GAN推理工具')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--no_save_recon', action='store_true', help='不保存重建结果')
    args = parser.parse_args()

    # 加载配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 创建推理器
    inferencer = Inferencer(config, Path(args.checkpoint))

    # 执行推理
    inferencer.process_batch(
        Path(args.input_dir),
        Path(args.output_dir),
        save_recon=not args.no_save_recon
    )


if __name__ == '__main__':
    main()