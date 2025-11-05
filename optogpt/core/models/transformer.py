"""
Transformer模型构建模块

本文件实现了完整的Transformer架构，包括编码器和解码器组件。
主要用于光学多层薄膜结构的正向设计（结构→光谱）和逆向设计（光谱→结构）。

主要组件：
- 编码器（Encoder）：处理输入序列
- 解码器（Decoder）：生成输出序列  
- 多头注意力机制（Multi-Head Attention）
- 位置编码（Positional Encoding）
- 前馈神经网络（Feed Forward Network）
"""

# Build models
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


def clones(module, N):
    """
    创建N个相同的层（模块）
    
    使用深拷贝确保每个模块的权重是独立的，不共享参数。
    这在构建多层Transformer时非常重要。
    
    参数：
        module: 要复制的PyTorch模块
        N: 复制的数量
        
    返回：
        nn.ModuleList: 包含N个独立模块的列表
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Embeddings(nn.Module):
    """
    词嵌入层
    
    将离散的词索引转换为连续的向量表示。
    
    属性：
        lut (nn.Embedding): 词嵌入查找表
        d_model (int): 嵌入向量的维度
    """
    def __init__(self, d_model, vocab):
        """
        初始化词嵌入层
        
        参数：
            d_model: 嵌入向量的维度
            vocab: 词汇表大小
        """
        super(Embeddings, self).__init__() 
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入的词索引张量
            
        返回：
            嵌入向量，乘以sqrt(d_model)进行缩放
        """
        # 返回x的嵌入向量，乘以sqrt(d_model)以稳定训练
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码层
    
    为输入序列添加位置信息，使模型能够利用序列中元素的位置关系。
    使用正弦和余弦函数生成位置编码。
    
    属性：
        dropout (nn.Dropout): dropout层
        pe (Tensor): 预计算的位置编码矩阵
    """
    def __init__(self, d_model, dropout, max_len=5000):
        """
        初始化位置编码层
        
        参数：
            d_model: 模型的维度
            dropout: dropout率
            max_len: 支持的最大序列长度
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
        pe_pos   = torch.mul(position, div_term)
        # 偶数位置使用sin，奇数位置使用cos
        pe[:, 0::2] = torch.sin(pe_pos)
        pe[:, 1::2] = torch.cos(pe_pos)
        pe = pe.unsqueeze(0)                                   
        # 将位置编码注册为buffer，不作为模型参数进行训练
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量
            
        返回：
            添加位置编码后的张量
        """
        # 根据输入序列长度截取相应的位置编码并添加到输入中
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    """
    计算缩放点积注意力（Scaled Dot Product Attention）
    
    注意力机制的核心计算：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k))V
    
    参数：
        query: 查询矩阵 Q
        key: 键矩阵 K
        value: 值矩阵 V
        mask: 可选的掩码矩阵，用于屏蔽某些位置
        dropout: 可选的dropout层
        
    返回：
        注意力加权后的值和注意力权重
    """
    # 获取查询向量的最后一维大小（即 d_k）
    d_k = query.size(-1) 
    # 计算注意力分数：QK^T / sqrt(d_k)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 如果提供了掩码，将被掩盖位置的分数设置为极小值
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # 对分数应用softmax得到注意力权重
    p_attn = F.softmax(scores, dim=-1)
    # 应用dropout（如果提供）
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 返回注意力加权的值和注意力权重
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    多头注意力机制
    
    将输入投影到多个子空间，在每个子空间并行计算注意力，然后合并结果。
    这使模型能够同时关注不同位置的不同表示子空间的信息。
    
    属性：
        h (int): 注意力头的数量
        d_k (int): 每个注意力头的维度
        linears (ModuleList): 4个线性变换层（WQ, WK, WV, WO）
        attn (Tensor): 注意力权重（用于可视化）
        dropout (nn.Dropout): dropout层
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        初始化多头注意力层
        
        参数：
            h: 注意力头的数量
            d_model: 模型的总维度
            dropout: dropout率
        """
        super(MultiHeadedAttention, self).__init__()
        # 确保d_model可以被h整除
        assert d_model % h == 0
        # 计算每个头的维度
        self.d_k = d_model // h
        self.h = h
        # 创建4个线性层：WQ, WK, WV和最终的线性映射WO
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        前向传播：应用多头注意力机制
        
        参数：
            query: 查询张量
            key: 键张量
            value: 值张量
            mask: 可选的掩码
            
        返回：
            多头注意力的输出
        """
        if mask is not None:
            # 对所有h个头应用相同的掩码
            mask = mask.unsqueeze(1)
        # 获取批次大小
        nbatches = query.size(0)

        # 1) 对Q、K、V进行线性投影，并重塑为(batch, h, seq_len, d_k)
        # 将d_model维度分割为h个头，每个头的维度为d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) 
                             for l, x in zip(self.linears, (query, key, value))]

        # 2) 对所有投影向量批量应用注意力机制
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) 使用view函数"拼接"多个头的输出，并应用最终的线性层
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # contiguous()：当使用transpose时，PyTorch不会创建新张量，只是改变元数据
        # 使用contiguous()创建一个带有转置数据的连续内存副本

        return self.linears[-1](x)  # 应用最终的线性层WO

class LayerNorm(nn.Module):
    """
    层归一化（Layer Normalization）
    
    对每个样本的特征进行归一化，有助于稳定训练过程。
    
    属性：
        a_2 (Parameter): 可学习的缩放参数
        b_2 (Parameter): 可学习的偏移参数
        eps (float): 用于数值稳定性的小常数
    """
    def __init__(self, features, eps=1e-6):
        """
        初始化层归一化
        
        参数：
            features: 特征维度
            eps: 防止除零的小常数
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量
            
        返回：
            归一化后的张量
        """
        # 计算每行的均值和标准差
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # 进行z-score标准化，然后应用可学习的缩放和偏移
        x_zscore = (x - mean)/ torch.sqrt(std ** 2 + self.eps) 
        return self.a_2*x_zscore+self.b_2 

class SublayerConnection(nn.Module):
    """
    残差连接后接层归一化
    
    用于连接多头注意力层和前馈网络层。
    采用Pre-LN结构：先归一化再应用子层，最后添加残差连接。
    
    属性：
        norm (LayerNorm): 层归一化
        dropout (nn.Dropout): dropout层
    """
    def __init__(self, size, dropout):
        """
        初始化子层连接
        
        参数：
            size: 层的大小
            dropout: dropout率
        """
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        对任何具有相同大小的子层应用残差连接
        
        参数：
            x: 输入张量
            sublayer: 子层函数
            
        返回：
            应用残差连接后的输出
        """
        # 先归一化，再应用子层，最后添加残差连接和dropout
        return x + self.dropout(sublayer(self.norm(x)))

class FullyConnectedLayers(nn.Module):
    """
    全连接层
    
    用于将编码器输出映射到目标维度。
    包含两个线性层和一个层归一化。
    
    属性：
        fc1 (nn.Linear): 第一个全连接层
        fc2 (nn.Linear): 第二个全连接层
        norm (LayerNorm): 层归一化
    """
    def __init__(self, input_dim, out_dim):
        """
        初始化全连接层
        
        参数：
            input_dim: 输入维度
            out_dim: 输出维度
        """
        super(FullyConnectedLayers, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, out_dim)
        self.norm = LayerNorm(input_dim)
    
    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量
            
        返回：
            全连接层的输出
        """
        # 第一层全连接 -> 归一化 -> 第二层全连接
        return self.fc2(self.norm(self.fc1(x)))


class PositionwiseFeedForward(nn.Module):
    """
    位置独立的前馈网络
    
    对序列中每个位置独立应用相同的两层全连接网络。
    
    属性：
        w_1 (nn.Linear): 第一个线性层
        w_2 (nn.Linear): 第二个线性层
        dropout (nn.Dropout): dropout层
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        初始化前馈网络
        
        参数：
            d_model: 模型的维度
            d_ff: 前馈网络的隐藏层维度
            dropout: dropout率
        """
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 输入张量
            
        返回：
            前馈网络的输出
        """
        # 第一层线性变换 -> dropout -> 第二层线性变换
        h1 = self.w_1(x)
        h2 = self.dropout(h1)
        return self.w_2(h2)

class Encoder(nn.Module):
    """
    编码器
    
    由N个相同的编码器层堆叠而成。
    
    属性：
        layers (ModuleList): N个编码器层的列表
        norm (LayerNorm): 最终的层归一化
    """
    def __init__(self, layer, N):
        """
        初始化编码器
        
        参数：
            layer: 单个编码器层
            N: 编码器层的数量
        """
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        依次通过每个编码器层处理输入
        
        参数：
            x: 输入张量
            mask: 掩码张量
            
        返回：
            编码器的输出
        """
        # 依次通过每个编码器层
        for layer in self.layers:
            x = layer(x, mask)
        # 最终进行层归一化
        return self.norm(x)

class EncoderLayer(nn.Module):
    """
    编码器层
    
    每个编码器层包含两个子层：
    1. 多头自注意力机制
    2. 位置独立的前馈网络
    每个子层都使用残差连接和层归一化。
    
    属性：
        self_attn (MultiHeadedAttention): 多头自注意力层
        feed_forward (PositionwiseFeedForward): 前馈网络
        sublayer (ModuleList): 两个子层连接
        size (int): 模型维度 d_model
    """
    def __init__(self, size, self_attn, feed_forward, dropout):
        """
        初始化编码器层
        
        参数：
            size: 模型维度 d_model
            self_attn: 多头自注意力模块
            feed_forward: 前馈网络模块
            dropout: dropout率
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """
        前向传播
        
        参数：
            x: 输入嵌入
            mask: 掩码
            
        返回：
            编码器层的输出
        """
        # 第一个子层：多头自注意力，使用lambda函数传递注意力计算
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 第二个子层：前馈网络
        return self.sublayer[1](x, self.feed_forward)

class Transformer(nn.Module):
    """
    Transformer模型（仅编码器版本）
    
    用于正向设计任务：从结构序列预测光谱。
    
    属性：
        encoder (Encoder): 编码器
        fc (FullyConnectedLayers): 全连接层，用于输出
        src_embed (Sequential): 源序列嵌入（包含词嵌入和位置编码）
    """
    def __init__(self, encoder, fc, src_embed):
        """
        初始化Transformer模型
        
        参数：
            encoder: 编码器模块
            fc: 全连接层
            src_embed: 源序列嵌入模块
        """
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.fc = fc
        self.src_embed = src_embed

    def encode(self, src, src_mask):
        """
        编码源序列
        
        参数：
            src: 源序列
            src_mask: 源序列掩码
            
        返回：
            编码器的输出
        """
        return self.encoder(self.src_embed(src), src_mask)

    def forward(self, src, src_mask):
        """
        前向传播
        
        参数：
            src: 源序列（结构）
            src_mask: 源序列掩码
            
        返回：
            预测的光谱
        """
        # 编码源序列
        en = self.encode(src, src_mask)
        # 只取第一个位置的输出（CLS token风格）
        en = en[:, 0,:]
        # 通过全连接层生成最终输出
        return self.fc(en)

def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1):
    """
    构建完整的Transformer模型（仅编码器版本，用于正向设计）
    
    参数：
        src_vocab: 源词汇表大小（结构词汇表）
        tgt_vocab: 目标词汇表大小（光谱维度）
        N: Transformer堆叠层数
        d_model: Query、Key、Value的维度
        d_ff: 前馈网络的神经元数量
        h: 注意力头的数量
        dropout: dropout率
        
    返回：
        初始化好的Transformer模型
    """
    c = copy.deepcopy
    # 创建多头注意力模块
    attn = MultiHeadedAttention(h, d_model)
    # 创建前馈网络模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 创建位置编码模块
    position = PositionalEncoding(d_model, dropout)
    # 创建全连接层
    fc = FullyConnectedLayers(d_model, tgt_vocab)
    # 构建完整的Transformer模型
    model = Transformer(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        fc, 
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)))
    
    # 使用Glorot/Xavier初始化参数
    # 参考论文：Understanding the difficulty of training deep feedforward neural networks
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


class Decoder(nn.Module):
    """
    解码器
    
    由N个相同的解码器层堆叠而成，带有掩码机制。
    
    属性：
        layers (ModuleList): N个解码器层的列表
        norm (LayerNorm): 最终的层归一化
    """
    def __init__(self, layer, N):
        """
        初始化解码器
        
        参数：
            layer: 单个解码器层
            N: 解码器层的数量
        """
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        重复N次解码器层
        
        参数：
            x: 目标序列的嵌入
            memory: 编码器的输出（记忆）
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码（包含subsequent mask）
            
        返回：
            解码器的输出
        """
        # 依次通过每个解码器层
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        # 最终进行层归一化
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    解码器层
    
    每个解码器层包含三个子层：
    1. 目标序列的多头自注意力
    2. 编码器-解码器多头注意力（交叉注意力）
    3. 位置独立的前馈网络
    每个子层都使用残差连接和层归一化。
    
    属性：
        size (int): 模型维度
        self_attn (MultiHeadedAttention): 自注意力层
        src_attn (MultiHeadedAttention): 交叉注意力层
        feed_forward (PositionwiseFeedForward): 前馈网络
        sublayer (ModuleList): 三个子层连接
    """
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        """
        初始化解码器层
        
        参数：
            size: 模型维度
            self_attn: 自注意力模块
            src_attn: 交叉注意力模块
            feed_forward: 前馈网络模块
            dropout: dropout率
        """
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        """
        前向传播
        
        参数：
            x: 目标序列嵌入
            memory: 编码器输出
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        返回：
            解码器层的输出
        """
        m = memory  # 编码器输出作为记忆
        # 第一个子层：目标序列的自注意力
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 第二个子层：交叉注意力（q来自解码器，k和v来自编码器）
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        # 第三个子层：前馈网络
        return self.sublayer[2](x, self.feed_forward)


def subsequent_mask(size):
    """
    生成subsequent mask，用于屏蔽未来位置
    
    在解码器的自注意力中，每个位置只能关注该位置之前的位置，
    不能看到未来的信息。这个函数生成一个上三角掩码矩阵。
    
    参数：
        size: 序列长度
        
    返回：
        下三角掩码矩阵（未来位置为False）
    """
    attn_shape = (1, size, size)
    # 创建上三角矩阵（k=1表示主对角线上方）
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回下三角矩阵（上三角为0的位置返回True）
    return torch.from_numpy(subsequent_mask) == 0

class Transformer_I(nn.Module):
    """
    Transformer逆向设计模型
    
    用于逆向设计任务：从光谱预测结构序列。
    包含全连接层处理光谱输入，解码器生成结构序列。
    
    属性：
        fc (FullyConnectedLayers): 全连接层，处理光谱输入
        decoder (Decoder): 解码器
        tgt_embed (Sequential): 目标序列嵌入（包含词嵌入和位置编码）
        generator (Generator): 生成器，将解码器输出转换为词汇表概率分布
    """
    def __init__(self, fc, decoder, tgt_embed, generator):
        """
        初始化逆向设计Transformer
        
        参数：
            fc: 全连接层
            decoder: 解码器模块
            tgt_embed: 目标序列嵌入模块
            generator: 生成器模块
        """
        super(Transformer_I, self).__init__()
        self.fc = fc
        self.decoder = decoder
        self.tgt_embed = tgt_embed
        self.generator = generator

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        解码函数
        
        参数：
            memory: 编码器输出（这里是处理后的光谱特征）
            src_mask: 源序列掩码
            tgt: 目标序列（结构）
            tgt_mask: 目标序列掩码
            
        返回：
            解码器的输出
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        前向传播
        
        参数：
            src: 源序列（光谱）
            tgt: 目标序列（结构）
            src_mask: 源序列掩码
            tgt_mask: 目标序列掩码
            
        返回：
            解码器的输出
        """
        # 通过全连接层处理光谱，然后解码生成结构
        return self.decode(self.fc(src), src_mask, tgt, tgt_mask)

class Generator(nn.Module):
    """
    生成器
    
    将解码器的输出转换为词汇表上的概率分布。
    
    属性：
        proj (nn.Linear): 线性投影层，从d_model映射到词汇表大小
    """
    def __init__(self, d_model, vocab):
        """
        初始化生成器
        
        参数：
            d_model: 模型维度
            vocab: 词汇表大小
        """
        super(Generator, self).__init__()
        # 解码：从d_model维度映射到词汇表大小
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
        前向传播
        
        参数：
            x: 解码器的输出
            
        返回：
            词汇表上的对数概率分布
        """
        # 应用线性投影和log_softmax
        return F.log_softmax(self.proj(x), dim=-1)

def make_model_I(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h = 8, dropout=0.1):
    """
    构建完整的Transformer逆向设计模型
    
    用于光谱到结构的逆向设计任务。
    
    参数：
        src_vocab: 源词汇表大小（光谱维度）
        tgt_vocab: 目标词汇表大小（结构词汇表）
        N: Transformer堆叠层数
        d_model: Query、Key、Value的维度
        d_ff: 前馈网络的神经元数量
        h: 注意力头的数量
        dropout: dropout率
        
    返回：
        初始化好的Transformer逆向设计模型
    """
    c = copy.deepcopy
    # 创建多头注意力模块
    attn = MultiHeadedAttention(h, d_model)
    # 创建前馈网络模块
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    # 创建位置编码模块
    position = PositionalEncoding(d_model, dropout)
    # 创建全连接层，将光谱特征映射到d_model维度
    fc = FullyConnectedLayers(src_vocab, d_model)
    # 构建完整的Transformer逆向设计模型
    model = Transformer_I(
        fc,
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab))
    
    # 使用Glorot/Xavier初始化参数
    # 参考论文：Understanding the difficulty of training deep feedforward neural networks
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
