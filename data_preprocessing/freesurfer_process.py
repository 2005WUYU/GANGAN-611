#!/usr/bin/env python3
"""
自动化调用FreeSurfer完成MRI数据处理
输入：原始T1加权MRI图像(.nii/.mgz)
输出：标准化的FreeSurfer输出目录结构
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Union, List, Optional
import argparse


class FreeSurferProcessor:
    def __init__(self,
                 subjects_dir: Union[str, Path],
                 n_jobs: int = 1,
                 log_level: str = "INFO"):
        """
        初始化FreeSurfer处理器

        Args:
            subjects_dir: FreeSurfer subjects目录路径
            n_jobs: 并行处理的作业数
            log_level: 日志级别
        """
        self.subjects_dir = Path(subjects_dir)
        self.n_jobs = n_jobs

        # 设置FreeSurfer环境变量
        os.environ['SUBJECTS_DIR'] = str(self.subjects_dir)

        # 配置日志
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # 检查FreeSurfer环境
        if not os.getenv('FREESURFER_HOME'):
            raise EnvironmentError("FREESURFER_HOME环境变量未设置")

    def process_subject(self,
                        t1_file: Union[str, Path],
                        subject_id: str,
                        reconall_args: Optional[List[str]] = None) -> bool:
        """
        处理单个被试的T1图像

        Args:
            t1_file: T1图像文件路径
            subject_id: 被试ID
            reconall_args: 额外的recon-all参数

        Returns:
            bool: 处理是否成功
        """
        t1_file = Path(t1_file)
        if not t1_file.exists():
            self.logger.error(f"T1文件不存在: {t1_file}")
            return False

        # 构建recon-all命令
        cmd = [
            'recon-all',
            '-subject', subject_id,
            '-i', str(t1_file),
            '-all'
        ]
        if reconall_args:
            cmd.extend(reconall_args)

        try:
            self.logger.info(f"开始处理被试 {subject_id}")
            subprocess.run(cmd, check=True)
            self.logger.info(f"成功完成被试 {subject_id} 的处理")
            return True
        except subprocess.CalledProcessError as e:
            self.logger.error(f"处理被试 {subject_id} 时发生错误: {e}")
            return False

    def verify_output(self, subject_id: str) -> bool:
        """
        验证FreeSurfer输出文件的完整性

        Args:
            subject_id: 被试ID

        Returns:
            bool: 输出是否完整
        """
        required_files = [
            f'surf/lh.pial',
            f'surf/rh.pial',
            f'surf/lh.curv',
            f'surf/rh.curv',
            f'surf/lh.sulc',
            f'surf/rh.sulc',
            f'surf/lh.thickness',
            f'surf/rh.thickness',
            f'label/lh.aparc.annot',
            f'label/rh.aparc.annot'
        ]

        subject_dir = self.subjects_dir / subject_id

        for file in required_files:
            if not (subject_dir / file).exists():
                self.logger.error(f"缺失文件: {file}")
                return False
        return True

    def batch_process(self,
                      subject_list: List[tuple],
                      skip_existing: bool = True) -> dict:
        """
        批量处理多个被试

        Args:
            subject_list: [(t1_file, subject_id), ...] 的列表
            skip_existing: 是否跳过已存在的被试

        Returns:
            dict: 处理结果统计
        """
        results = {
            'success': [],
            'failed': [],
            'skipped': []
        }

        for t1_file, subject_id in subject_list:
            if skip_existing and (self.subjects_dir / subject_id).exists():
                self.logger.info(f"跳过已存在的被试: {subject_id}")
                results['skipped'].append(subject_id)
                continue

            if self.process_subject(t1_file, subject_id):
                if self.verify_output(subject_id):
                    results['success'].append(subject_id)
                else:
                    results['failed'].append(subject_id)
            else:
                results['failed'].append(subject_id)

        self.logger.info(f"处理完成。成功: {len(results['success'])}, "
                         f"失败: {len(results['failed'])}, "
                         f"跳过: {len(results['skipped'])}")
        return results


def main():
    parser = argparse.ArgumentParser(description='FreeSurfer批处理工具')
    parser.add_argument('--subjects_dir', required=True, help='FreeSurfer subjects目录')
    parser.add_argument('--input_list', required=True, help='输入文件列表(CSV格式:t1_file,subject_id)')
    parser.add_argument('--n_jobs', type=int, default=1, help='并行作业数')
    args = parser.parse_args()

    # 读取输入列表
    with open(args.input_list) as f:
        subject_list = [tuple(line.strip().split(',')) for line in f]

    processor = FreeSurferProcessor(args.subjects_dir, args.n_jobs)
    processor.batch_process(subject_list)


if __name__ == '__main__':
    main()