"""
训练主脚本
- 模型训练循环
- 损失计算
- 模型保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from tqdm import tqdm

from models.generator import Generator
from models.discriminator import PatchDiscriminator
from training.dataset import BrainGraphDataset, create_dataloader
from training.losses import GANLoss, SpectralLoss, l1_loss


class Trainer:
    def __init__(self,
                 config: Dict[str, Any],
                 device: torch.device = torch.device('cuda')):
        """
        训练器
        Args:
            config: 配置字典
            device: 计算设备
        """
        self.config = config
        self.device = device

        # 设置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 创建输出目录
        self.output_dir = Path(config['training']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 保存配置
        with open(self.output_dir / 'config.yaml', 'w') as f:
            yaml.dump(config, f)

        # 初始化模型
        self._init_models()

        # 初始化数据集
        self._init_datasets()

        # 初始化损失函数
        self._init_losses()

        # 初始化优化器
        self._init_optimizers()

        # 初始化TensorBoard
        self.writer = SummaryWriter(self.output_dir / 'logs')

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

    def _init_datasets(self):
        """初始化数据集"""
        data_config = self.config['data']

        # 创建数据集
        train_dataset = BrainGraphDataset(
            root_dir=data_config['data_dir'],
            split='train',
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio']
        )

        val_dataset = BrainGraphDataset(
            root_dir=data_config['data_dir'],
            split='val',
            train_ratio=data_config['train_ratio'],
            val_ratio=data_config['val_ratio']
        )

        # 创建DataLoader
        self.train_loader = create_dataloader(
            train_dataset,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )

        self.val_loader = create_dataloader(
            val_dataset,
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers']
        )

    def _init_losses(self):
        """初始化损失函数"""
        loss_config = self.config['loss']

        self.gan_loss = GANLoss(
            lambda_gp=loss_config['lambda_gp'],
            device=self.device
        )

        self.spectral_loss = SpectralLoss(
            k=loss_config['spectral_k']
        )

        self.lambda_recon = loss_config['lambda_recon']
        self.lambda_spec = loss_config['lambda_spec']

    def _init_optimizers(self):
        """初始化优化器"""
        optim_config = self.config['optimizer']

        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=optim_config['g_lr'],
            betas=(optim_config['beta1'], optim_config['beta2'])
        )

        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=optim_config['d_lr'],
            betas=(optim_config['beta1'], optim_config['beta2'])
        )

    def train_step(self, batch: Any) -> Dict[str, float]:
        """单步训练"""
        # 将数据移至GPU
        batch = batch.to(self.device)

        # 训练判别器
        for _ in range(self.config['training']['n_critic']):
            self.d_optimizer.zero_grad()

            # 生成重建图
            fake = self.generator(batch.x, batch.edge_index, batch.edge_attr)

            # 判别器前向传播
            real_scores = self.discriminator(
                batch.x, batch.edge_index, batch.labels
            )
            fake_scores = self.discriminator(
                fake.detach(), batch.edge_index, batch.labels
            )

            # 判别器损失
            d_loss, gp = self.gan_loss.discriminator_loss(
                real_scores, fake_scores, batch.x, fake, self.discriminator
            )

            d_loss.backward()
            self.d_optimizer.step()

        # 训练生成器
        self.g_optimizer.zero_grad()

        # 重新生成（因为前面的fake已经被detach）
        fake = self.generator(batch.x, batch.edge_index, batch.edge_attr)

        # 对抗损失
        fake_scores = self.discriminator(
            fake, batch.edge_index, batch.labels
        )
        g_loss_adv = self.gan_loss.generator_loss(fake_scores)

        # 重建损失
        g_loss_recon = l1_loss(fake, batch.x)

        # 谱损失
        g_loss_spec = self.spectral_loss(
            batch.edge_index, batch.edge_index,
            batch.x.size(0)
        )

        # 总损失
        g_loss = (g_loss_adv +
                  self.lambda_recon * g_loss_recon +
                  self.lambda_spec * g_loss_spec)

        g_loss.backward()
        self.g_optimizer.step()

        return {
            'd_loss': d_loss.item(),
            'gp': gp.item(),
            'g_loss': g_loss.item(),
            'g_loss_adv': g_loss_adv.item(),
            'g_loss_recon': g_loss_recon.item(),
            'g_loss_spec': g_loss_spec.item()
        }

    def validate(self) -> Dict[str, float]:
        """验证"""
        self.generator.eval()
        self.discriminator.eval()

        val_metrics = {
            'val_recon_loss': 0.0,
            'val_spec_loss': 0.0
        }

        with torch.no_grad():
            for batch in self.val_loader:
                batch = batch.to(self.device)
                fake = self.generator(batch.x, batch.edge_index, batch.edge_attr)

                val_metrics['val_recon_loss'] += l1_loss(fake, batch.x).item()
                val_metrics['val_spec_loss'] += self.spectral_loss(
                    batch.edge_index, batch.edge_index,
                    batch.x.size(0)
                ).item()

        # 计算平均值
        for k in val_metrics:
            val_metrics[k] /= len(self.val_loader)

        return val_metrics

    def train(self):
        """训练循环"""
        train_config = self.config['training']
        best_val_loss = float('inf')

        for epoch in range(train_config['n_epochs']):
            self.generator.train()
            self.discriminator.train()

            # 训练一个epoch
            epoch_metrics = {
                'd_loss': 0.0,
                'gp': 0.0,
                'g_loss': 0.0,
                'g_loss_adv': 0.0,
                'g_loss_recon': 0.0,
                'g_loss_spec': 0.0
            }

            pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
            for batch in pbar:
                step_metrics = self.train_step(batch)

                # 更新进度条
                pbar.set_postfix(step_metrics)

                # 累积指标
                for k, v in step_metrics.items():
                    epoch_metrics[k] += v

            # 计算平均值
            for k in epoch_metrics:
                epoch_metrics[k] /= len(self.train_loader)

            # 验证
            val_metrics = self.validate()

            # 记录指标
            for k, v in {**epoch_metrics, **val_metrics}.items():
                self.writer.add_scalar(k, v, epoch)

            # 保存最佳模型
            val_loss = val_metrics['val_recon_loss']
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(
                    self.output_dir / 'best_model.pt',
                    epoch, val_loss
                )

            # 保存最新模型
            if (epoch + 1) % train_config['save_freq'] == 0:
                self.save_checkpoint(
                    self.output_dir / f'epoch_{epoch + 1}.pt',
                    epoch, val_loss
                )

            self.logger.info(
                f'Epoch {epoch}: ' +
                ' '.join([f'{k}={v:.4f}' for k, v in {**epoch_metrics, **val_metrics}.items()])
            )

    def save_checkpoint(self,
                        path: Path,
                        epoch: int,
                        val_loss: float):
        """保存检查点"""
        torch.save({
            'epoch': epoch,
            'val_loss': val_loss,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict()
        }, path)

    def load_checkpoint(self, path: Path):
        """加载检查点"""
        checkpoint = torch.load(path)

        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])

        return checkpoint['epoch'], checkpoint['val_loss']


def main():
    """主函数"""
    # 加载配置
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 创建训练器
    trainer = Trainer(config)

    # 开始训练
    trainer.train()


if __name__ == '__main__':
    main()