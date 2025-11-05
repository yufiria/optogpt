"""
训练模块

本文件包含训练Transformer模型的核心功能，包括：
- 损失函数（标签平滑）
- 优化器（Noam学习率调度）
- 训练循环
- 模型保存和加载

支持两种模式：
- 正向设计训练（结构→光谱）
- 逆向设计训练（光谱→结构）
"""

import os
import math
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk import word_tokenize
from collections import Counter
from torch.autograd import Variable
import seaborn as sns
import matplotlib.pyplot as plt
import pickle as pkl
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist



class LabelSmoothing(nn.Module):
    """
    标签平滑损失函数
    
    标签平滑是一种正则化技术，通过将真实标签的概率分布
    从one-hot编码软化为更平滑的分布来防止过拟合。
    
    属性：
        criterion (nn.KLDivLoss): KL散度损失
        padding_idx (int): 填充token的索引
        confidence (float): 真实标签的置信度
        smoothing (float): 平滑参数
        size (int): 词汇表大小
        true_dist (Tensor): 平滑后的真实分布
    """
    def __init__(self, size, padding_idx, smoothing=0.0):
        """
        初始化标签平滑
        
        参数：
            size: 词汇表大小
            padding_idx: 填充token的索引
            smoothing: 平滑系数（0表示无平滑）
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        """
        前向传播
        
        参数：
            x: 模型预测的对数概率 (batch_size, vocab_size)
            target: 真实标签索引 (batch_size,)
            
        返回：
            KL散度损失
        """
        assert x.size(1) == self.size
        # 创建平滑后的真实分布
        true_dist = x.data.clone()
        # 将平滑概率均匀分配给非真实标签（除了PAD和UNK）
        true_dist.fill_(self.smoothing / (self.size - 2))
        # 将真实标签位置设置为较高的置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # 填充位置的概率设为0
        true_dist[:, self.padding_idx] = 0
        # 找出目标为填充的位置
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            # 将这些位置的整行概率设为0
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # 计算KL散度
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    """
    简单的损失计算类
    
    用于逆向设计任务的损失计算和反向传播。
    
    属性：
        generator (Generator): 生成器模块
        criterion (LabelSmoothing): 损失函数
        opt (NoamOpt): 优化器（可选）
    """
    def __init__(self, generator, criterion, opt=None):
        """
        初始化损失计算
        
        参数：
            generator: 生成器模块
            criterion: 损失函数
            opt: 优化器（训练时提供，验证时为None）
        """
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):
        """
        计算损失并执行反向传播
        
        参数：
            x: 解码器输出
            y: 真实标签
            norm: 归一化因子（非填充token数量）
            
        返回：
            归一化后的损失值
        """
        # 通过生成器得到词汇表上的概率分布
        x = self.generator(x)
        # 重塑张量以计算损失
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                              y.contiguous().view(-1)) / norm
        # 反向传播
        loss.backward()
        # 如果提供了优化器，更新参数
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        # 返回未归一化的损失值
        return loss.data.item() * norm.float()

# Noam学习率调度器示例参数：factor=2, warmup-step = 4000
def get_std_opt(model):
    """
    获取标准的Noam优化器
    
    参数：
        model: 模型
        
    返回：
        配置好的NoamOpt优化器
    """
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class NoamOpt:
    """
    Noam学习率调度器
    
    实现Transformer论文中的学习率调度策略：
    lr = factor * (model_size^-0.5 * min(step^-0.5, step * warmup^-1.5))
    
    包含warmup阶段，学习率先线性增长，然后按步数的平方根衰减。
    
    属性：
        optimizer (Optimizer): PyTorch优化器
        _step (int): 当前步数
        warmup (int): warmup步数
        factor (float): 学习率缩放因子
        model_size (int): 模型维度
        _rate (float): 当前学习率
    """
    def __init__(self, model_size, factor, warmup, optimizer):
        """
        初始化Noam优化器
        
        参数：
            model_size: 模型维度
            factor: 学习率缩放因子
            warmup: warmup步数
            optimizer: 底层优化器
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        """
        更新参数和学习率
        """
        self._step += 1
        rate = self.rate()
        # 更新优化器的学习率
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        # 执行优化步骤
        self.optimizer.step()
        
    def rate(self, step = None):
        """
        计算当前学习率
        
        参数：
            step: 指定步数（默认使用当前步数）
            
        返回：
            计算得到的学习率
        """
        if step is None:
            step = self._step
        # Noam学习率公式
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))


def run_epoch(data, model, criterion, optimizer, epoch, DEVICE):
    """
    运行一个训练/验证epoch（正向设计模式）
    
    参数：
        data: 批次数据列表
        model: Transformer模型
        criterion: 损失函数
        optimizer: 优化器（验证时为None）
        epoch: 当前epoch编号
        DEVICE: 计算设备
        
    返回：
        平均损失
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    
    for i , batch in enumerate(data):
        # 前向传播
        out = model(batch.src.to(DEVICE),  batch.src_mask.to(DEVICE))
        # 计算损失
        loss = criterion(out, batch.trg.to(DEVICE))

        # 如果提供了优化器，执行反向传播和参数更新
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            optimizer.optimizer.zero_grad()

        # 累计损失和token数
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # 每50个批次打印一次进度
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(
                epoch, i - 1, loss, (tokens.float() / elapsed)))
            start = time.time()
            tokens = 0
        del out, loss
    
    print(total_loss, i)
    return total_loss/i

def count_params(model):
    """
    计算模型的参数数量
    
    参数：
        model: PyTorch模型
        
    返回：
        可训练参数的总数
    """
    return sum([np.prod(layer.size()) for layer in model.parameters() if layer.requires_grad])

def save_checkpoint(model, optimizer, epoch, loss_all, path, configs):
    """
    保存模型检查点
    
    参数：
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss_all: 损失历史记录
        path: 保存路径
        configs: 配置参数
    """
    # 保存模型状态、优化器状态和训练信息
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer,
            'loss_all':loss_all,
            'configs':configs,
        }, path)


def train(data, model, criterion, optimizer, configs, DEVICE):
    """
    训练模型（正向设计模式）
    
    执行完整的训练循环，包括训练和验证，保存最佳模型。
    
    参数：
        data: PrepareData对象，包含训练和验证数据
        model: Transformer模型
        criterion: 损失函数
        optimizer: 优化器
        configs: 配置参数对象
        DEVICE: 计算设备
    """
    # 初始化最佳验证损失为一个很大的值
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # 训练模式
        model.train()
        train_loss = run_epoch(data.train_data, model, criterion, optimizer, epoch, DEVICE)

        # 验证模式
        model.eval()
        print('>>>>> Evaluate')
        with torch.no_grad():
            dev_loss = run_epoch(data.dev_data, model, criterion, None, epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.8f}'.format(dev_loss))
        
        # 记录损失
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())

        # 如果验证损失降低，保存最佳模型
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 
                          'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_best.pt',  configs)
            print('Saved')
        
        # 每2个epoch保存一次最近的模型
        if epoch%2 == 1:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 
                          'saved_models/ol_transformer/'+save_folder+'/'+save_name+'_recent.pt',  configs)
            
        print(f">>>>> current best loss: ", best_dev_loss)


def run_epoch_I(data, model, loss_compute, epoch, DEVICE):
    """
    运行一个训练/验证epoch（逆向设计模式）
    
    参数：
        data: 批次数据列表
        model: Transformer逆向设计模型
        loss_compute: SimpleLossCompute对象
        epoch: 当前epoch编号
        DEVICE: 计算设备
        
    返回：
        每个token的平均损失
    """
    start = time.time()
    total_tokens = 0.
    total_loss = 0.
    tokens = 0.
    
    for i , batch in enumerate(data):
        # 前向传播
        out = model(batch.src.to(DEVICE), batch.trg.to(DEVICE), 
                   batch.src_mask, batch.trg_mask.to(DEVICE))
        # 计算损失并执行反向传播
        loss = loss_compute(out, batch.trg_y.to(DEVICE), batch.ntokens.to(DEVICE))
        
        # 累计损失和token数
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        
        # 每50个批次打印一次进度
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch {:d} Batch: {:d} Loss: {:.4f} Tokens per Sec: {:.2f}s".format(
                epoch, i - 1, loss / batch.ntokens, (tokens.float() / elapsed)))
            start = time.time()
            tokens = 0
        del out, loss

    # 返回每个token的平均损失
    return total_loss / total_tokens

    
def train_I(data, model, criterion, optimizer, configs, DEVICE):
    """
    训练模型（逆向设计模式）
    
    执行完整的训练循环，用于光谱到结构的逆向设计任务。
    
    参数：
        data: PrepareData对象，包含训练和验证数据
        model: Transformer逆向设计模型
        criterion: 标签平滑损失函数
        optimizer: Noam优化器
        configs: 配置参数对象
        DEVICE: 计算设备
    """
    # 初始化最佳验证损失为一个很大的值
    best_dev_loss = 1e5
    loss_all = {'train_loss':[], 'dev_loss':[]}

    save_folder = configs.save_folder
    save_name = configs.save_name
    EPOCHS = configs.epochs

    for epoch in range(EPOCHS):
        # 训练模式
        model.train()
        train_loss = run_epoch_I(data.train_data, model, 
                                 SimpleLossCompute(model.generator, criterion, optimizer), 
                                 epoch, DEVICE)
        
        # 验证模式
        model.eval()
        print('>>>>> Evaluate')
        dev_loss = run_epoch_I(data.dev_data, model, 
                              SimpleLossCompute(model.generator, criterion, None), 
                              epoch, DEVICE)
        print('<<<<< Evaluate loss: {:.2f}'.format(dev_loss))
        
        # 记录损失
        loss_all['train_loss'].append(train_loss.detach())
        loss_all['dev_loss'].append(dev_loss.detach())
        
        # 如果验证损失降低，保存最佳模型
        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            save_checkpoint(model, optimizer, epoch, loss_all, 
                          'saved_models/optogpt/'+save_folder+'/'+save_name+'_best.pt',  configs)

        # 每个epoch都保存最近的模型
        save_checkpoint(model, optimizer, epoch, loss_all, 
                      'saved_models/optogpt/'+save_folder+'/'+save_name+'_recent.pt',  configs)
            
        print(f">>>>> current best loss: {best_dev_loss}")
        