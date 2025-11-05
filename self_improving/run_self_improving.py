#!/usr/bin/env python
"""
自我改进数据增强流程主程序

本脚本演示了完整的自我改进数据增强工作流，无需硬编码路径。

主要步骤：
1. 生成开发数据
2. 准备增强数据
3. 选择需要扰动的数据
4. 扰动数据（使用GA、PSO等方法）
5. 模拟扰动后的结构
6. 过滤改进的数据
7. 合并原始数据和增强数据
8. 重新训练模型

使用方法：
    python run_self_improving.py --model_path path/to/model.pt \
        --train_struct_path path/to/train_struct.pkl \
        --train_spec_path path/to/train_spec.pkl \
        --dev_struct_path path/to/dev_struct.pkl \
        --dev_spec_path path/to/dev_spec.pkl
"""

import argparse
import os
import torch
import pandas as pd
from pathlib import Path

from prepare_aug_data import Prepare_Augment_Data
from data_perturb import get_to_be_perturbed_data, perturb_data, simulate_perturbed_struct, get_perturbed_better_data
from combine_data import combine_data
from model_retrain import retrain_model
from generate_dev_data import generate_dev_data
from core.datasets.datasets import PrepareData


def parse_args():
    """
    解析命令行参数
    
    返回：
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(description='OptoGPT自我改进数据增强')
    
    # 必需路径参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='预训练OptoGPT模型路径')
    parser.add_argument('--train_struct_path', type=str, required=True,
                        help='训练集结构数据路径（pkl文件）')
    parser.add_argument('--train_spec_path', type=str, required=True,
                        help='训练集光谱数据路径（pkl文件）')
    parser.add_argument('--dev_struct_path', type=str, required=True,
                        help='验证集结构数据路径（pkl文件）')
    parser.add_argument('--dev_spec_path', type=str, required=True,
                        help='验证集光谱数据路径（pkl文件）')
    
    # 输出目录
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='输出保存目录')
    
    # 方法配置
    parser.add_argument('--decoding_method', type=str, default='TOP-KP Decode_v2',
                        choices=['Greedy Decode', 'TOP-KP Decode', 'TOP-KP Decode_v2', 'Beam Search'],
                        help='解码方法')
    parser.add_argument('--perturbation_method', type=str, default='GA_PSO',
                        choices=['random', 'PSO', 'GA_PSO'],
                        help='扰动方法')
    parser.add_argument('--error_type', type=str, default='MSE',
                        choices=['MAE', 'MSE'],
                        help='误差度量类型')
    
    # 超参数
    parser.add_argument('--give_up_threshold', type=float, default=3.0,
                        help='结构的最大误差阈值')
    parser.add_argument('--kp_num', type=int, default=50,
                        help='每个光谱的解码尝试次数')
    parser.add_argument('--keep_num', type=int, default=20,
                        help='保留的最佳结构数量')
    parser.add_argument('--target_aug_size', type=int, default=200000,
                        help='增强数据集的目标大小')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='重训练轮数')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='早停耐心值')
    
    # GPU设置
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备ID（-1表示CPU）')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='多进程的CPU工作进程数')
    
    return parser.parse_args()


def setup_directories(output_dir):
    """
    创建必要的输出目录
    
    参数：
        output_dir: 输出根目录
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'augmented_data'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)


def main():
    """
    主函数：执行完整的自我改进数据增强流程
    """
    args = parse_args()
    
    # 设置设备
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu}')
    else:
        device = torch.device('cpu')
    
    # 在模块中覆盖硬编码的设备设置
    import prepare_aug_data
    import data_perturb
    prepare_aug_data.DEVICE = device
    data_perturb.DEVICE = device
    
    # 设置多进程工作进程
    import multiprocessing
    if args.num_workers > 0:
        multiprocessing.set_start_method('spawn', force=True)
    
    # 创建目录
    setup_directories(args.output_dir)
    
    print(f"运行自我改进数据增强...")
    print(f"设备: {device}")
    print(f"输出目录: {args.output_dir}")
    
    # 步骤1: 生成开发数据
    print("\n1. 生成开发数据...")
    dev_spec = generate_dev_data()
    print(f"生成了 {len(dev_spec)} 个开发光谱")
    
    # 步骤2: 准备增强数据
    print("\n2. 准备增强数据...")
    augment_data = Prepare_Augment_Data(
        model_path=args.model_path,
        decoding_method=args.decoding_method,
        error_type=args.error_type,
        top_k=10,
        top_p=0.9,
        kp_num=args.kp_num,
        keep_num=args.keep_num
    )
    
    # 步骤3: 获取需要扰动的数据
    print("\n3. 选择需要扰动的数据...")
    perturbed_df = get_to_be_perturbed_data(augment_data, args.give_up_threshold)
    print(f"选择了 {len(perturbed_df)} 个结构进行扰动")
    
    # 步骤4: 扰动数据
    print(f"\n4. 使用 {args.perturbation_method} 方法扰动数据...")
    perturbed_df = perturb_data(perturbed_df, method=args.perturbation_method)
    
    # 保存中间结果
    perturbed_df.to_pickle(os.path.join(args.output_dir, 'augmented_data', 'perturbed_data.pkl'))
    
    # 步骤5: 模拟扰动后的结构
    print("\n5. 模拟扰动后的结构...")
    perturbed_df = simulate_perturbed_struct(perturbed_df, error_type=args.error_type)
    
    # 步骤6: 过滤改进的数据
    print("\n6. 过滤改进的结构...")
    added_data = get_perturbed_better_data(perturbed_df)
    initial_size = len(added_data)
    print(f"找到 {initial_size} 个改进的结构")
    
    # 删除重复项
    added_data = added_data.drop_duplicates(subset=['new_error'])
    
    # 复制以达到目标大小
    while len(added_data) < args.target_aug_size:
        added_data = pd.concat([added_data, added_data], ignore_index=True)
    added_data = added_data[:args.target_aug_size]
    
    # 保存增强数据
    added_data.to_pickle(os.path.join(args.output_dir, 'augmented_data', 'added_data.pkl'))
    print(f"增强数据集大小: {len(added_data)}")
    
    # 步骤7: 合并数据
    print("\n7. 合并原始数据和增强数据...")
    new_train_spec_path, new_train_struct_path, new_test_spec_path, new_test_struct_path = combine_data(
        args.train_spec_path,
        args.train_struct_path,
        args.dev_spec_path,
        args.dev_struct_path,
        added_data,
        ratio=0.1,
        type_T=args.decoding_method
    )
    
    # 将合并后的数据移动到输出目录
    import shutil
    for old_path, new_name in [
        (new_train_spec_path, 'train_spec_augmented.pkl'),
        (new_train_struct_path, 'train_struct_augmented.pkl'),
        (new_test_spec_path, 'test_spec_augmented.pkl'),
        (new_test_struct_path, 'test_struct_augmented.pkl')
    ]:
        new_path = os.path.join(args.output_dir, 'augmented_data', new_name)
        shutil.move(old_path, new_path)
        if 'train_spec' in new_name:
            new_train_spec_path = new_path
        elif 'train_struct' in new_name:
            new_train_struct_path = new_path
        elif 'test_spec' in new_name:
            new_test_spec_path = new_path
        elif 'test_struct' in new_name:
            new_test_struct_path = new_path
    
    # 步骤8: 加载原始模型配置
    print("\n8. 加载模型配置...")
    model_checkpoint = torch.load(args.model_path, map_location=device)
    original_args = model_checkpoint['configs']
    
    # 从原始数据获取词汇表
    data = PrepareData(
        args.train_struct_path, 
        args.train_spec_path, 
        original_args.ratios, 
        args.dev_struct_path, 
        args.dev_spec_path, 
        original_args.batch_size, 
        original_args.spec_type, 
        'Inverse'
    )
    struct_word_dict = data.struc_word_dict
    struct_index_dict = data.struc_index_dict
    
    # 步骤9: 重新训练模型
    print(f"\n9. 重新训练模型，训练 {args.epochs} 轮...")
    
    # 在重训练函数中覆盖设备选择
    import model_retrain
    model_retrain.DEVICE = device
    
    # 模型目录
    models_dir = os.path.join(args.output_dir, 'models')
    
    retrain_model(
        args.model_path,
        new_train_struct_path,
        new_train_spec_path,
        new_test_spec_path,
        new_test_struct_path,
        args.epochs,
        args.early_stopping_patience,
        struct_word_dict,
        struct_index_dict,
        args.decoding_method,
        device=device,
        output_dir=models_dir
    )
    
    print(f"\n✓ 自我改进数据增强完成！")
    print(f"结果保存在: {args.output_dir}")
    print(f"- 增强数据: {os.path.join(args.output_dir, 'augmented_data')}")
    print(f"- 训练模型: {models_dir}")


if __name__ == '__main__':
    main() 