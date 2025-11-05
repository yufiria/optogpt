"""
数据集处理模块

本文件负责处理光学多层薄膜的数据集，包括：
- 数据加载和预处理
- 词汇表构建
- 批处理和填充
- 数据掩码生成

主要功能：
- 从pkl文件加载结构和光谱数据
- 将结构转换为词ID序列
- 生成训练和验证数据批次
"""

import numpy as np
import torch
from collections import Counter
import pickle as pkl
from torch.autograd import Variable

# 特殊token的ID
UNK = 0  # 未知词ID
PAD = 1  # 填充词ID

def seq_padding(X, padding=0):
    """
    对批次数据进行填充，使所有序列长度一致
    
    将批次中的所有序列填充到最长序列的长度，
    短序列在末尾用padding值填充。
    
    参数：
        X: 序列列表，每个序列可能有不同的长度
        padding: 填充值，默认为0
        
    返回：
        填充后的numpy数组，所有序列长度相同
    """
    # 获取所有序列的长度
    L = [len(x) for x in X]
    # 找到最大长度
    ML = max(L)
    # 对每个序列进行填充
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])    
    
class Batch:
    """
    批次数据对象
    
    用于在训练期间保存带有掩码的批次数据。
    支持正向设计（结构→光谱）和逆向设计（光谱→结构）两种模式。
    
    属性：
        src: 源序列张量
        trg: 目标序列张量（正向）或目标序列（逆向，不包含最后一个token）
        trg_y: 目标标签（逆向模式，不包含第一个token）
        src_mask: 源序列掩码
        trg_mask: 目标序列掩码
        ntokens: 非填充token的总数
    """
    def __init__(self, src, trg, if_inverse = 'Forward', pad=0):
        """
        初始化批次对象
        
        参数：
            src: 源序列数组
            trg: 目标序列数组
            if_inverse: 'Forward'表示正向设计，'Inverse'表示逆向设计
            pad: 填充值
        """
        # 将词ID转换为long格式
        src = torch.from_numpy(src).long()
        trg = torch.tensor(trg).float()
        
        if if_inverse == 'Forward':
            # 正向模式：源是结构，目标是光谱
            self.src = src
            self.trg = trg
            # 生成填充位置的二值掩码
            # 将矩阵形状改为1×序列长度
            self.src_mask = (src != pad).unsqueeze(-2)
            self.ntokens = (self.src != pad).data.sum()
        elif if_inverse == 'Inverse':
            # 逆向模式：源是光谱，目标是结构
            # 解码器的输入：去掉最后一个token
            self.trg = src[:, :-1]
            # 解码器的目标：去掉第一个token（用于计算损失）
            self.trg_y = src[:, 1:]
            # 源序列（光谱）增加一个维度
            self.src = trg.unsqueeze(-2)
            # 光谱不需要掩码
            self.src_mask = None
            # 生成目标序列掩码（包含subsequent mask）
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg != pad).data.sum()
        else:
            raise NotImplementedError
            
    @staticmethod
    def make_std_mask(tgt, pad):
        """
        创建掩码以隐藏填充和未来的词
        
        参数：
            tgt: 目标序列
            pad: 填充值
            
        返回：
            组合的掩码（填充掩码 & subsequent掩码）
        """
        # 创建填充掩码
        tgt_mask = (tgt != pad).unsqueeze(-2)
        # 与subsequent mask结合（定义在decoder部分）
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

def subsequent_mask(size):
    """
    生成subsequent mask，用于屏蔽未来位置
    
    在解码时，每个位置只能看到该位置之前的信息。
    
    参数：
        size: 序列长度
        
    返回：
        下三角掩码矩阵
    """
    attn_shape = (1, size, size)
    # 创建上三角矩阵
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # 返回下三角矩阵（True表示可见）
    return torch.from_numpy(subsequent_mask) == 0


class PrepareData:
    """
    数据准备类
    
    负责加载、预处理和批处理数据。
    主要步骤：
    1. 加载数据并进行分词
    2. 构建词汇表
    3. 将词转换为ID
    4. 生成批次数据
    
    属性：
        train_struc: 训练集结构数据
        train_spec: 训练集光谱数据
        dev_struc: 验证集结构数据
        dev_spec: 验证集光谱数据
        struc_word_dict: 结构词到ID的字典
        struc_index_dict: ID到结构词的字典
        struc_total_words: 词汇表总大小
        train_data: 训练批次列表
        dev_data: 验证批次列表
    """
    def __init__(self, train_file, train_spec_file, train_ratio, dev_file, dev_spec_file, BATCH_SIZE=128, spec_type = 'R_T', if_inverse = 'Forward'):
        """
        初始化数据准备类
        
        参数：
            train_file: 训练集结构文件路径
            train_spec_file: 训练集光谱文件路径
            train_ratio: 训练数据使用比例（0-100）
            dev_file: 验证集结构文件路径
            dev_spec_file: 验证集光谱文件路径
            BATCH_SIZE: 批次大小
            spec_type: 光谱类型，'R'(反射)/'T'(透射)/'R_T'(两者)
            if_inverse: 'Forward'或'Inverse'，指定任务类型
        """
        # 01. 读取数据并进行分词
        self.train_struc, self.train_spec = self.load_data(train_file, train_spec_file, train_ratio)
        self.dev_struc, self.dev_spec = self.load_data(dev_file, dev_spec_file)

        # 根据光谱类型选择相应的光谱数据
        dims = self.train_spec.shape[1]//2
        if spec_type == 'R':
            # 只使用反射光谱
            self.train_spec = self.train_spec[:, :dims]
            self.dev_spec = self.dev_spec[:, :dims]
        elif spec_type == 'T':
            # 只使用透射光谱
            self.train_spec = self.train_spec[:, dims:]
            self.dev_spec = self.dev_spec[:, dims:]

        # 02. 构建词典：结构
        self.struc_word_dict, self.struc_total_words, self.struc_index_dict = self.build_dict(self.train_struc)

        # 03. 使用词典将词转换为ID
        self.train_struc = self.wordToID(self.train_struc, self.struc_word_dict)
        self.dev_struc = self.wordToID(self.dev_struc, self.struc_word_dict)

        # 04. 生成批次 + 填充 + 掩码
        self.train_data = self.splitBatch(self.train_struc, self.train_spec, BATCH_SIZE, if_inverse)
        self.dev_data   = self.splitBatch(self.dev_struc, self.dev_spec, BATCH_SIZE, if_inverse)

    def load_data(self, path, spec_path, ratio = 100):
        """
        读取结构和光谱数据
        
        对结构进行分词并添加开始/结束标记（BOS/EOS）。
        
        参数：
            path: 结构数据文件路径（pkl格式）
            spec_path: 光谱数据文件路径（pkl格式）
            ratio: 使用数据的百分比，范围(0, 100]
            
        返回：
            struc: 处理后的结构列表
            spec: 光谱数据数组
        """
        struc = []
        all_struc = []

        # 加载结构数据
        with open (path, 'rb') as fp:
            all_struc = pkl.load(fp)

        # 为每个结构添加BOS和EOS标记
        for ele in all_struc:
            struc.append(["BOS"] + ele + ["EOS"])

        # 加载光谱数据
        with open (spec_path, 'rb') as fp:
            spec = pkl.load(fp)       

        # 检查ratio参数的有效性
        if ratio <=0 or ratio > 100:
            raise NameError('Wrong training dataset ratio. Make sure it is (0, 100]. ') 
        
        # 根据ratio截取数据
        lengs = len(struc)
        struc = struc[:int(ratio*lengs/100)]
        spec = spec[:int(ratio*lengs/100)]

        return struc, spec
    
    def build_dict(self, sentences, max_words = 1000):
        """
        构建词汇表字典
        
        统计词频并创建词到ID的映射字典。
        
        参数：
            sentences: 词列表的列表
            max_words: 词汇表最大大小
            
        返回：
            word_dict: 词到ID的字典 {词: ID}
            total_words: 词汇表总大小
            index_dict: ID到词的字典 {ID: 词}
        """
        # 统计词频
        word_count = Counter()
        for sentence in sentences:
            for s in sentence:
                word_count[s] += 1

        # 获取最常见的max_words个词
        ls = word_count.most_common(max_words)
        total_words = len(ls) + 2  # +2是为了UNK和PAD
        
        # 构建词到ID的字典（ID从2开始，0和1留给UNK和PAD）
        word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
        word_dict['UNK'] = UNK
        word_dict['PAD'] = PAD
        
        # 构建ID到词的反向字典
        index_dict = {v: k for k, v in word_dict.items()}
        return word_dict, total_words, index_dict

    def wordToID(self, en, en_dict, sort=False):
        """
        将输入/输出词列表转换为ID列表
        
        参数：
            en: 词序列列表
            en_dict: 词到ID的字典
            sort: 是否按长度排序（减少填充）
            
        返回：
            out_en_ids: ID序列列表
        """
        # 将每个词转换为ID，未知词使用0
        out_en_ids = [[en_dict.get(w, 0) for w in sent] for sent in en]

        def len_argsort(seq):
            """
            获取按长度排序的索引
            
            参数：
                seq: 序列列表
                
            返回：
                排序后的索引列表
            """
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))

        # 如果需要排序，按序列长度重新排列
        if sort:
            sorted_index = len_argsort(out_en_ids)
            out_en_ids = [out_en_ids[id] for id in sorted_index]
        return out_en_ids

    def splitBatch(self, struc, spec, batch_size, if_inverse = 'Forward', shuffle=False):
        """
        将数据分割成批次
        
        参数：
            struc: 结构数据列表
            spec: 光谱数据列表
            batch_size: 批次大小
            if_inverse: 'Forward'或'Inverse'
            shuffle: 是否打乱数据
            
        返回：
            batches: Batch对象列表
        """
        # 生成批次起始索引
        idx_list = np.arange(0, len(struc), batch_size)
        if shuffle:
            np.random.shuffle(idx_list)

        # 为每个批次生成索引范围
        batch_indexs = []
        for idx in idx_list:
            batch_indexs.append(np.arange(idx, min(idx + batch_size, len(struc))))
        
        # 创建批次对象
        batches = []
        for batch_index in batch_indexs:
            # 获取当前批次的数据
            batch_struc = [struc[index] for index in batch_index]  
            batch_spec = [spec[index] for index in batch_index]

            # 对结构序列进行填充
            batch_struc = seq_padding(batch_struc)
            # 创建Batch对象
            batches.append(Batch(batch_struc, np.array(batch_spec), if_inverse)) 

        return batches
