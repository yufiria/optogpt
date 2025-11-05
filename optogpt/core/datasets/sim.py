"""
光学多层薄膜仿真模块

本文件使用传输矩阵法(TMM)进行光学多层薄膜的光谱仿真。
主要功能：
- 加载材料的折射率(n)和消光系数(k)数据
- 计算多层薄膜结构的反射率(R)和透射率(T)
- 支持相干和非相干层的计算

使用的物理方法：
- 传输矩阵法 (Transfer Matrix Method)
- CIE颜色匹配函数
"""

import numpy as np
from numpy import pi
import colour
import pandas as pd
import colour
import pickle as pkl
from tmm import coh_tmm, inc_tmm
from scipy.interpolate import interp1d
from colour import SDS_ILLUMINANTS, SpectralDistribution
from colour.colorimetry import MSDS_CMFS
from colour.plotting import plot_single_colour_swatch, ColourSwatch, plot_chromaticity_diagram_CIE1931
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib as mpl
import os
import itertools
from multiprocessing import Pool
import pyswarms as ps
from colour.difference import delta_E, delta_E_CIE2000


# 材料数据库路径
DATABASE = './nk'
# D65标准光源（日光）
illuminant = SDS_ILLUMINANTS['D65']
# CIE 1931 2度标准观察者颜色匹配函数
cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

# 可用材料列表
mats = ['Al', 'Al2O3', 'AlN', 'Ge', 'HfO2', 'ITO', 'MgF2', 'MgO', 'Si', 'Si3N4', 'SiO2', 'Ta2O5', 'TiN', 'TiO2', 'ZnO', 'ZnS', 'ZnSe', 'Glass_Substrate']
# 可用厚度列表（nm）
thicks = [str(i) for i in range(5, 255, 5)]

# 波长范围设置（微米）
lamda_low = 0.4
lamda_high = 1.1
wavelengths = np.arange(lamda_low, lamda_high+1e-3, 0.01)



def load_materials(all_mats = mats, wavelengths = wavelengths, DATABASE = './nk'):
    """
    加载材料的光学常数（折射率n和消光系数k）
    
    从CSV文件中读取材料数据，并使用插值函数将数据扩展到指定波长范围。
    
    参数：
        all_mats: 材料名称列表
        wavelengths: 波长数组（微米）
        DATABASE: 材料数据库路径
    
    返回：
        nk_dict: 字典，键为材料名称，值为复折射率数组 (n + ik)
    """
    nk_dict = {}

    for mat in all_mats:
        # 读取材料的n和k数据
        nk = pd.read_csv(os.path.join(DATABASE, mat + '.csv'))
        nk.dropna(inplace=True)

        wl = nk['wl'].to_numpy()
        index_n = nk['n'].to_numpy()
        index_k = nk['k'].to_numpy()

        # 使用三次样条插值n，线性插值k
        n_fn = interp1d(
                wl, index_n,  bounds_error=False, fill_value='extrapolate', kind=3)
        k_fn = interp1d(
                wl, index_k,  bounds_error=False, fill_value='extrapolate', kind=1)
        
        # 计算复折射率: n_complex = n + ik
        nk_dict[mat] = n_fn(wavelengths) + 1j*k_fn(wavelengths)

    return nk_dict

def spectrum(materials, thickness, pol = 's', theta=0,  wavelengths = wavelengths, nk_dict = {}, substrate = 'Glass_Substrate', substrate_thick = 500000):
    """
    计算多层薄膜结构的光谱（反射率和透射率）
    
    使用传输矩阵法计算非相干多层薄膜的光学响应。
    
    参数：
        materials: 材料名称列表（从顶层到底层）
        thickness: 对应的厚度列表（nm）
        pol: 偏振类型，'s'或'p'
        theta: 入射角（度）
        wavelengths: 波长数组（微米）
        nk_dict: 材料折射率字典
        substrate: 基底材料名称
        substrate_thick: 基底厚度（nm）
    
    返回：
        光谱数组：前半部分是反射率R，后半部分是透射率T
    """
    # 角度转弧度
    degree = pi/180
    theta = theta *degree
    # 波长转换为nm
    wavess = (1e3 * wavelengths).astype('int')

    # 构建厚度列表：[无穷大(空气), 薄膜层..., 基底, 无穷大(空气)]
    thickness = [np.inf] + thickness + [substrate_thick, np.inf]

    R, T, A = [], [], []
    # 设置相干性：'i'=非相干，'c'=相干
    # 空气层和基底为非相干，薄膜层为相干
    inc_list = ['i'] + ['c']*len(materials) + ['i', 'i']
    
    # 对每个波长进行计算
    for i, lambda_vac in enumerate(wavess):
        # 构建折射率列表：[空气, 薄膜层..., 基底, 空气]
        n_list = [1] + [nk_dict[mat][i] for mat in materials] + [nk_dict[substrate][i], 1]

        # 使用非相干传输矩阵法计算
        res = inc_tmm(pol, n_list, thickness, inc_list, theta, lambda_vac)

        R.append(res['R'])  # 反射率
        T.append(res['T'])  # 透射率

    # 返回拼接的R和T光谱
    return R + T