"""
通用工具模块

本文件包含一些通用的工具函数和类，主要用于：
- 内存监控和分析
- 数据集处理和序列化
- 进程内存使用情况跟踪

主要组件：
- MemoryMonitor：内存监控类
- DatasetFromList：从列表创建PyTorch数据集
- get_mem_info：获取进程内存信息
"""

from __future__ import annotations
from collections import defaultdict
import pickle
import sys
import torch
import json
from typing import Any
from tabulate import tabulate
import os
import time
import psutil


def get_mem_info(pid: int) -> dict[str, int]:
    """
    获取进程的内存信息
    
    收集进程的各种内存使用指标。
    
    参数：
        pid: 进程ID
    
    返回：
        包含内存使用信息的字典：
        - rss: 常驻集大小（实际物理内存）
        - pss: 比例集大小（共享内存按比例分配）
        - uss: 唯一集大小（进程独占内存）
        - shared: 共享内存总量
        - shared_file: 文件映射的共享内存
    """
    res = defaultdict(int)
    for mmap in psutil.Process(pid).memory_maps():
        res['rss'] += mmap.rss
        res['pss'] += mmap.pss
        res['uss'] += mmap.private_clean + mmap.private_dirty
        res['shared'] += mmap.shared_clean + mmap.shared_dirty
        if mmap.path.startswith('/'):
            res['shared_file'] += mmap.shared_clean + mmap.shared_dirty
    return res


class MemoryMonitor():
    """
    内存监控类
    
    用于监控和报告一个或多个进程的内存使用情况。
    
    属性：
        pids (list[int]): 要监控的进程ID列表
        data (dict): 存储最新的内存数据
    """
    def __init__(self, pids: list[int] = None):
        """
        初始化内存监控器
        
        参数：
            pids: 进程ID列表，默认为当前进程
        """
        if pids is None:
            pids = [os.getpid()]
        self.pids = pids

    def add_pid(self, pid: int):
        """
        添加要监控的进程
        
        参数：
            pid: 要添加的进程ID
        """
        assert pid not in self.pids
        self.pids.append(pid)

    def _refresh(self):
        """
        刷新内存数据
        
        返回：
            更新后的内存数据字典
        """
        self.data = {pid: get_mem_info(pid) for pid in self.pids}
        return self.data

    def table(self) -> str:
        """
        以表格形式返回内存使用情况
        
        返回：
            格式化的表格字符串
        """
        self._refresh()
        table = []
        keys = list(list(self.data.values())[0].keys())
        now = str(int(time.perf_counter() % 1e5))
        for pid, data in self.data.items():
            table.append((now, str(pid)) + tuple(self.format(data[k]) for k in keys))
        return tabulate(table, headers=["time", "PID"] + keys)

    def str(self):
        """
        以字符串形式返回内存使用情况
        
        返回：
            格式化的字符串
        """
        self._refresh()
        keys = list(list(self.data.values())[0].keys())
        res = []
        for pid in self.pids:
            s = f"PID={pid}"
            for k in keys:
                v = self.format(self.data[pid][k])
                s += f", {k}={v}"
            res.append(s)
        return "\n".join(res)

    @staticmethod
    def format(size: int) -> str:
        """
        格式化内存大小
        
        将字节数转换为人类可读的格式（B, KB, MB, GB）
        
        参数：
            size: 字节数
        
        返回：
            格式化的字符串
        """
        for unit in ('', 'K', 'M', 'G'):
            if size < 1024:
                break
            size /= 1024.0
        return "%.1f%s" % (size, unit)


def create_coco() -> list[Any]:
    """
    创建COCO数据集标注列表
    
    从COCO 2017训练集标注文件中加载数据。
    数据可从以下地址下载：
    https://huggingface.co/datasets/merve/coco/resolve/main/annotations/instances_train2017.json
    
    返回：
        COCO标注列表
    """
    with open("instances_train2017.json") as f:
        obj = json.load(f)
        return obj["annotations"]


def read_sample(x):
    """
    读取样本对象并增加其引用计数
    
    模拟真实数据加载器的行为。
    Python 3.10.6之前的版本中，pickle不会增加引用计数（这是一个bug）。
    该bug在 https://github.com/python/cpython/pull/92931 中修复。
    
    参数：
        x: 要序列化的对象
    
    返回：
        序列化后的数据
    """
    if sys.version_info >= (3, 10, 6):
        # 在此版本之后，pickle会正确增加引用计数
        return pickle.dumps(x)
    else:
        # 旧版本使用msgpack
        import msgpack
        return msgpack.dumps(x)


class DatasetFromList(torch.utils.data.Dataset):
    """
    从列表创建PyTorch数据集
    
    简单的数据集包装器，将列表转换为PyTorch Dataset对象。
    
    属性：
        lst (list): 数据列表
    """
    def __init__(self, lst):
        """
        初始化数据集
        
        参数：
            lst: 数据列表
        """
        self.lst = lst
        
    def __len__(self):
        """返回数据集长度"""
        return len(self.lst)
        
    def __getitem__(self, idx: int):
        """
        获取指定索引的数据项
        
        参数：
            idx: 数据索引
        
        返回：
            对应索引的数据项
        """
        return self.lst[idx]


# 主程序示例
if __name__ == "__main__":
    from serialize import NumpySerializedList
    
    # 创建内存监控器
    monitor = MemoryMonitor()
    print("Initial", monitor.str())
    
    # 加载COCO数据
    lst = create_coco()
    print("JSON", monitor.str())
    
    # 序列化数据
    lst = NumpySerializedList(lst)
    print("Serialized", monitor.str())
    
    # 删除并回收内存
    del lst
    import gc
    gc.collect()
    print("End", monitor.str())
