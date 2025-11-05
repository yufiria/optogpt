"""
OL-Transformer正向设计主程序

本文件是OL-Transformer正向设计任务的入口脚本，用于从多层薄膜结构预测光谱。
主要功能：
- 解析命令行参数
- 加载和准备数据集
- 构建Transformer正向设计模型（仅编码器）
- 训练模型并保存结果

使用方法：
    python run_ol_transformer.py --epochs 1000 --batch_size 1000 --spec_type R_T
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl

# 设置计算设备（优先使用CUDA）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 导入核心模块
from core.datasets.datasets import *
from core.models.transformer import *
from core.trains.train import *

print(DEVICE)

if __name__ == '__main__':
    
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 训练参数
    parser.add_argument('--seeds', default=42, type=int, help='随机种子')
    parser.add_argument('--epochs', default=1000, type=int, help='训练轮数')
    parser.add_argument('--ratios', default=100, type=int, help='训练数据集使用比例')
    parser.add_argument('--batch_size', default=1000, type=int, help='批次大小')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout率')
    parser.add_argument('--max_lr', default=1.0, type=float, help='最大学习率')
    parser.add_argument('--warm_steps', default=100000, type=int, help='学习率预热步数')

    # 数据维度参数
    parser.add_argument('--struc_dim', default=104, type=int, help='结构token数量')
    parser.add_argument('--spec_dim', default=142, type=int, help='光谱维度')

    # 模型架构参数
    parser.add_argument('--layers', default=12, type=int, help='编码器层数')
    parser.add_argument('--head_num', default=8, type=int, help='注意力头数量')
    parser.add_argument('--d_model', default=1024, type=int, help='模型总维度 = head_num * head_dim')
    parser.add_argument('--d_ff', default=512, type=int, help='前馈层维度')
    parser.add_argument('--max_len', default=22, type=int, help='Transformer最大序列长度')

    # 保存路径参数
    parser.add_argument('--save_folder', default='test', type=str, help='一级保存文件夹')
    parser.add_argument('--save_name', default='model_forward', type=str, help='模型保存名称')
    parser.add_argument('--spec_type', default='R_T', type=str, help='光谱类型：R/T/R_T')
    
    # 数据文件路径参数
    parser.add_argument('--TRAIN_FILE', default='TRAIN_FILE', type=str, help='训练结构文件')
    parser.add_argument('--TRAIN_SPEC_FILE', default='TRAIN_SPEC_FILE', type=str, help='训练光谱文件')
    parser.add_argument('--DEV_FILE', default='DEV_FILE', type=str, help='验证结构文件')
    parser.add_argument('--DEV_SPEC_FILE', default='DEV_SPEC_FILE', type=str, help='验证光谱文件')
    
    # 词典参数
    parser.add_argument('--struc_index_dict', default={2:'BOS'}, type=dict, help='结构索引到词的字典')
    parser.add_argument('--struc_word_dict', default={'BOS':2}, type=dict, help='结构词到索引的字典')

    args = parser.parse_args()

    # 设置随机种子以保证可重复性
    torch.manual_seed(args.seeds)
    np.random.seed(args.seeds)

    # 构建保存文件名，包含关键超参数
    temp = [args.ratios, args.batch_size, args.max_lr, args.warm_steps, 
            args.layers, args.head_num, args.d_model, args.d_ff]
    args.save_name += '_' + args.spec_type
    args.save_name += '_R_B_LR_WU_L_H_D_F_'+str(temp)

    # 设置数据文件路径
    TRAIN_FILE = './dataset/Structure_train.pkl'   
    TRAIN_SPEC_FILE = './dataset/Spectrum_train.pkl'  
    DEV_FILE = './dataset/Structure_dev.pkl'   
    DEV_SPEC_FILE = './dataset/Spectrum_dev.pkl'  

    args.TRAIN_FILE, args.TRAIN_SPEC_FILE, args.DEV_FILE, args.DEV_SPEC_FILE = TRAIN_FILE, TRAIN_SPEC_FILE, DEV_FILE, DEV_SPEC_FILE

    # 准备数据（正向设计模式）
    data = PrepareData(TRAIN_FILE, TRAIN_SPEC_FILE, args.ratios, DEV_FILE, DEV_SPEC_FILE, 
                      args.batch_size, args.spec_type)

    # 获取词汇表大小和光谱维度
    src_vocab = len(data.struc_word_dict)  # 源词汇表（结构）
    tgt_vocab = len(data.dev_spec[0])      # 目标维度（光谱）
    args.struc_dim = src_vocab
    args.spec_dim = tgt_vocab
    args.struc_index_dict = data.struc_index_dict
    args.struc_word_dict = data.struc_word_dict

    print(f"struc_vocab {src_vocab}")
    print(f"spec_vocab {tgt_vocab}")

    # 构建正向设计模型
    model = make_model(
                    args.struc_dim,    # 源：结构词汇表大小
                    args.spec_dim,     # 目标：光谱维度
                    args.layers, 
                    args.d_model, 
                    args.d_ff,
                    args.head_num,
                    args.dropout
                ).to(DEVICE)

    print('Model Transformer, Number of parameters {}'.format(count_params(model)))

    # 开始训练
    print(">>>>>>> start train")
    train_start = time.time()
    
    # 使用MSE损失函数（回归任务）
    criterion = torch.nn.MSELoss()

    # 创建Noam优化器
    optimizer = NoamOpt(args.d_model, args.max_lr, args.warm_steps, 
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9,0.98), eps=1e-9))

    # 执行训练
    train(data, model, criterion, optimizer, args, DEVICE)
    print(f"<<<<<<< finished train, cost {time.time()-train_start:.4f} seconds")







