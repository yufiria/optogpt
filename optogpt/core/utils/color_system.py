"""
颜色系统模块

本文件实现了CIE颜色系统，用于将光谱转换为RGB颜色值。
主要功能：
- 定义颜色系统的三原色和白点
- 光谱到XYZ颜色空间的转换
- XYZ到RGB颜色空间的转换
- RGB到HTML十六进制颜色的转换

基于CIE 1931标准观察者颜色匹配函数。
"""

import numpy as np

def xyz_from_xy(x, y):
    """
    从xy色度坐标计算XYZ向量
    
    在CIE色度图中，z = 1 - x - y
    
    参数：
        x: x色度坐标
        y: y色度坐标
    
    返回：
        (x, y, z)向量，其中z = 1-x-y
    """
    return np.array((x, y, 1-x-y))

class ColourSystem:
    """
    颜色系统类
    
    通过CIE x, y和z=1-x-y坐标定义的颜色系统，包括三原色和白点。
    用于光谱到RGB颜色的转换。
    
    TODO: 实现gamma校正
    
    属性：
        cmf (ndarray): CIE颜色匹配函数，400-780nm，5nm间隔
        red (ndarray): 红色原色的xyz坐标
        green (ndarray): 绿色原色的xyz坐标
        blue (ndarray): 蓝色原色的xyz坐标
        white (ndarray): 白点的xyz坐标
        M (ndarray): 色度矩阵 (rgb -> xyz)
        MI (ndarray): M的逆矩阵
        wscale (ndarray): 白点缩放数组
        T (ndarray): xyz -> rgb 转换矩阵
    """

    # CIE颜色匹配函数，400-780 nm，5 nm间隔
    cmf = np.loadtxt('cie-cmf.txt', usecols=(1,2,3))

    def __init__(self, red, green, blue, white):
        """
        初始化颜色系统对象
        
        为定义颜色系统的每个红、绿、蓝原色色度和白点光源传递向量
        （即形状为(3,)的NumPy数组）。
        
        参数：
            red: 红色原色的色度向量
            green: 绿色原色的色度向量
            blue: 蓝色原色的色度向量
            white: 白点光源的向量
        """
        # 色度坐标
        self.red, self.green, self.blue = red, green, blue
        self.white = white
        
        # 色度矩阵 (rgb -> xyz) 及其逆矩阵
        self.M = np.vstack((self.red, self.green, self.blue)).T 
        self.MI = np.linalg.inv(self.M)
        
        # 白点缩放数组
        self.wscale = self.MI.dot(self.white)
        
        # xyz -> rgb 转换矩阵
        self.T = self.MI / self.wscale[:, np.newaxis]

    def xyz_to_rgb(self, xyz, out_fmt=None):
        """
        将xyz色彩空间转换为rgb表示
        
        输出的rgb分量在其最大值上归一化。如果xyz超出rgb色域，
        将其去饱和直到进入色域。
        
        参数：
            xyz: xyz颜色向量
            out_fmt: 输出格式，默认返回小数rgb；'html'返回十六进制字符串'#rrggbb'
        
        返回：
            rgb向量或HTML十六进制字符串
        """
        # 转换为rgb
        rgb = self.T.dot(xyz)
        
        # 如果rgb分量不全为零，归一化到最大值
        if not np.all(rgb==0):
            rgb /= np.max(rgb)

        # 如果需要HTML格式，转换为十六进制字符串
        if out_fmt == 'html':
            return self.rgb_to_hex(rgb)
        return rgb

    def rgb_to_hex(self, rgb):
        """
        将小数rgb值转换为HTML样式的十六进制字符串
        
        参数：
            rgb: 范围在[0,1]的rgb值数组
        
        返回：
            '#rrggbb'格式的十六进制字符串
        """
        hex_rgb = (255 * rgb).astype(int)
        return '#{:02x}{:02x}{:02x}'.format(*hex_rgb)

    def spec_to_xyz(self, spec):
        """
        将光谱转换为xyz颜色点
        
        光谱必须与颜色匹配函数在相同的网格点上：380-780 nm，5 nm间隔。
        
        参数：
            spec: 光谱强度数组
        
        返回：
            归一化的xyz颜色坐标
        """
        # 将光谱与颜色匹配函数相乘并求和得到XYZ
        XYZ = np.sum(spec[:, np.newaxis] * self.cmf, axis=0)
        den = np.sum(XYZ)
        # 如果总和为0，直接返回
        if den == 0.:
            return XYZ
        # 归一化XYZ
        return XYZ / den

    def spec_to_rgb(self, spec, out_fmt=None):
        """
        将光谱转换为rgb值
        
        参数：
            spec: 光谱强度数组
            out_fmt: 输出格式，默认返回小数rgb；'html'返回十六进制字符串
        
        返回：
            rgb向量或HTML十六进制字符串
        """
        # 先转换为xyz，再转换为rgb
        xyz = self.spec_to_xyz(spec)
        return self.xyz_to_rgb(xyz, out_fmt)

# D65标准光源（代表日光）
illuminant_D65 = xyz_from_xy(0.3127, 0.3291)

# sRGB颜色系统（标准RGB）
cs_srgb = ColourSystem(red=xyz_from_xy(0.64, 0.33),
                       green=xyz_from_xy(0.30, 0.60),
                       blue=xyz_from_xy(0.15, 0.06),
                       white=illuminant_D65)