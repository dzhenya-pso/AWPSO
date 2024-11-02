# -*- coding: utf-8 -*-
"""
Created on Tue Dec 05 09:36:15 2023
@author:Slogan
About:
"""
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
# from CEC2013benchmarkfunctions.opfunu.CEC13 import CEC13  # 导入测试函数集

np.random.seed(42)


class AWPSO():
    def __init__(self, fitness, fbias, fun_num=1, D=30, P=20, G=500, ub=1, lb=0,
                 w_max=0.9, w_min=0.4, c1=2.0, c2=2.0, k=0.2):
        self.fitness = fitness  # 适应度
        self.D = D  # 搜索空间维数
        self.P = P  # 种群规模
        self.G = G  # 最大迭代次数
        self.ub = ub * np.ones([self.P, self.D])  # 上限
        self.lb = lb * np.ones([self.P, self.D])  # 下限
        self.fbias = fbias
        self.num = fun_num
        self.w_max = w_max  # 最大惯性权重
        self.w_min = w_min  # 最小惯性权重
        self.w = w_max  # 初始时惯性权重最大
        self.c1 = c1  # 加速因子1
        self.c2 = c2  # 加速因子2
        self.k = k
        self.v_max = self.k * (self.ub - self.lb)  # 初始化最大速度

        self.pbest_X = np.zeros([self.P, self.D])  # 初始化局部最佳位置
        self.pbest_F = np.zeros([self.P]) + np.inf  # 初始化局部最佳位置的适应度
        self.gbest_X = np.zeros([self.D])  # 初始化全局最佳位置
        self.gbest_F = np.inf  # 初始化全局最佳位置的适应度
        self.loss_curve = np.zeros(self.G)  # 最佳适应度

    def opt(self):
        # 初始化
        self.X = np.random.uniform(low=self.lb, high=self.ub, size=[self.P, self.D])  # 随机初始化粒子群位置
        self.V = np.zeros([self.P, self.D])  # 随机初始化粒子群速度

        # # 迭代输出,进度条格式
        # for self.g, iterm in zip(range(0, self.G),
        #                          tqdm(list(range(0, self.G)),
        #                               desc="\033[92mProcessing\033[93m",
        #                               unit="\033[94m iterm\033[94m", colour='blue')):
        # 普通输出
        for self.g in range(self.G):
            # 计算每个粒子的适应度值
            self.F = self.fitness(self.X) - self.fbias
            # self.F = CEC13(self.X, self.num) - self.fbias

            # 更新最佳解
            mask = self.F < self.pbest_F  # 如果当前粒子的适应度好于局部最佳粒子
            self.pbest_X[mask] = self.X[mask].copy()  # 则将其作为局部最佳位置（历史最佳）
            self.pbest_F[mask] = self.F[mask].copy()

            if np.min(self.F) < self.gbest_F:  # 如果当前最佳粒子的适应度好于全局最佳粒子
                idx = self.F.argmin()
                self.gbest_X = self.X[idx].copy()  # 则将其作为全局最佳位置
                self.gbest_F = self.F.min()

            # 收敛曲线
            self.loss_curve[self.g] = self.gbest_F
            # Update the W of PSO
            self.w = self.w_max - ((self.w_max - self.w_min) * self.g) / self.G
            # 更新
            r1 = np.random.uniform(size=[self.P, self.D])
            r2 = np.random.uniform(size=[self.P, self.D])
            self.c1 = self.F_c1()
            self.c2 = self.F_c2()

            self.V = self.w * self.V + self.c1 * (self.pbest_X - self.X) * r1 \
                     + self.c2 * (self.gbest_X - self.X) * r2  # 公式（1）
            self.V = np.clip(self.V, -self.v_max, self.v_max)  # 边界处理

            self.X = self.X + self.V  # 公式（2）
            self.X = np.clip(self.X, self.lb, self.ub)  # 边界处理

    def plot_curve(self):
        plt.figure()
        plt.title('plot curve [' + str(round(self.loss_curve[-1], 3)) + ']')
        plt.plot(self.loss_curve, label=f'CEC2017fun_{self.num} plot')
        plt.grid()
        plt.legend()
        plt.show()

    def F_c1(self):
        a = 0.0000035 * (self.ub - self.lb)
        b = 0.5
        c = 0
        d = 1.5
        result = b / (1 + np.exp(-a * ((self.pbest_X - self.X) - c))) + d
        return result

    def F_c2(self):
        a = 0.0000035 * (self.ub - self.lb)
        b = 0.5
        c = 0
        d = 1.5
        result = b / (1 + np.exp(-a * ((self.gbest_X - self.X) - c))) + d
        return result