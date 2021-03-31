import random
import sys
import logging
import imp #import函数
import itertools
import numpy as np

import pandas as pd
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions
import pickle
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F                 # 导入torch.nn.functional
from PycharmProjects.MPC.est_mpc import picture_u,picture_m,picture_mfd1,picture_mfd2
np.random.seed(0)#使得随机数据可预测，如果不设置seed，则每次会生成不同的随机数
torch.manual_seed(0)#为CPU中设置种子，生成随机数，设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进
control_step=99
episode=100


#############产生高需求与低需求，两个场景###################
f0 = open(r'E:\wyy\PycharmProjects\MPC\temp_file\aa.pkl', 'rb')
bb = pickle.load(f0)
f0.close()
d=bb.d_2
d_low = bb.d_low_2
demand=d
#######################产生宏观控制参数U##################################
f2 = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\bb.pkl', 'rb')
bbb = pickle.load(f2)
f2.close()
M1_2,M2_1,U21,U12,x,N1,N2,m1,m2=bbb.M1_2,bbb.M2_1,bbb.U21,bbb.U12,bbb.x,bbb.N1,bbb.N2,bbb.m1,bbb.m2
# print(M1_2)
# print(M2_1)
picture_u(x,U21,U12)
picture_m(x,M2_1,M1_2)

#######################强化学习##################################
BATCH_SIZE = 32                                # 样本数量
LR = 0.02                                       # 学习率
EPSILON = 0.8                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 1000                           # 记忆库容量
# 定义环境

Action={0:[10,50,30,30],1:[15,45,30,30],2:[20,40,30,30],3:[25,35,30,30],4:[30,30,30,30],5:[35,25,30,30],6:[40,20,30,30],7:[45,15,30,30],
        8:[50,10,25,35],9:[30,30,25,35],10:[30,30,20,40],11:[30,30,15,45],12:[30,30,10,50],13:[30,30,35,25],14:[30,30,40,20],15:[30,30,45,15],
        16:[30,30,50,10],17:[30,20,40,30]}
N_ACTIONS = len(Action)                                 # 动作个数 (6个)
N_STATES = 18
# AC算法
imp.reload(logging)
logging.basicConfig(level=logging.DEBUG,
        format='%(asctime)s [%(levelname)s] %(message)s',
        stream=sys.stdout, datefmt='%H:%M:%S')
class Env(object):
    def __init__(self):
        self._max_episode_steps=10
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        #####初始化#####
        self.s_start=[]
        for t in range(1):
            for h in range(4):
                for i in self.n_m:
                       self.n_m[i][t][h]=5
        for h in range(4):
            for i in self.n_m:
                self.s_start.append(self.n_m[i][0][h])
        self.s_start.append(20)
        self.s_start.append(20)
        self.td_c  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 私家车时延
        self.i_mc= {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车与公交车之和
        self.sta_flow  = 0.4  # 饱和流
        self.l_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.s_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}
        self.trans_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.trans_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}
        self.trans_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}

        self.remain_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                          '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.remain_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                          '出口4': np.zeros((900, 4))}
        self.remain_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                          '出口4': np.zeros((900, 4))}
        self.actal_o_m = {'1-2': np.zeros((900, 4)), '2-1': np.zeros((900, 4))}
        self.ZhuanYi1to2=[]
        self.ZhuanYi2to1 = []

        self.o_m  =  {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.remain_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.qc1 = 0.05
        self.qc2 = 0.05
        self.C = 120
        self.vc = {'区域1': np.ones(900), '区域2': np.ones(900)}
    def reset(self):
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        #####初始化#####
        self.s_start = []
        for t in range(1):
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = 5
        for h in range(4):
            for i in self.n_m:
                self.s_start.append(self.n_m[i][0][h])
        self.s_start.append(20)
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
                    '出口4': np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        #####初始化#####
        self.s_start = []
        for t in range(1):
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = 5
        for h in range(4):
            for i in self.n_m:
                self.s_start.append(self.n_m[i][0][h])
        self.s_start.append(20)
        self.s_start.append(20)
        return self.s_start


    def DuiBi(self, l_xo, BHDl, l_TongXingNL,T=120):
        if l_xo >= BHDl:
            Q = 0
        else:
            Q = (T * l_TongXingNL / 4) *((BHDl - 1)+ math.sqrt((BHDl - 1)**2 + (12 * (BHDl - l_xo)) / (l_TongXingNL *T)))
        return Q

    def DDelay(self, BHD, action,a):
        LXB = action[a] / self.C
        if BHD >= 1:
            d1 = (self.C - action[a]) / 2
        else:
            d1 = self.C * (1 - LXB) ** 2 / (2 * (1 - LXB * BHD))
        return d1

    def delayy(self, h,action,a, zuozhuan,zhixing):
         T=120
         l_xo = self.sta_flow/4##饱和流率
         # print('######x参数',l_xo)
         l_TongXingNL = 1400/3600
         BHDl = (zuozhuan/self.C) / l_TongXingNL#饱和度
         # print('######饱和度', BHDl)

         Qcl = env.DuiBi(self, l_xo, BHDl, l_TongXingNL,T) / l_TongXingNL
         # print('######左转随机延误与过饱和延误', Qcl)
         Dl = env.DDelay(self, BHDl, action, a)
         # print('######均衡相位延误', Dl)
         delayl = Qcl + Dl

         s_xo = self.sta_flow/2##饱和流率
         # print('######s_xo参数', s_xo)
         s_TongXingNL = 1400*2 / 3600
         BHDs = (zhixing /self.C)/s_TongXingNL
         # print('######直行饱和度', BHDs)
         Qcs = (env.DuiBi(self, s_xo, BHDs, s_TongXingNL) )/ s_TongXingNL
         # print('######直行随机延误与过饱和延误', Qcs)
         Ds = env.DDelay(self, BHDs, action, a)
         delays = Qcs + Ds
         ddd = (delayl * zuozhuan + delays * zhixing)
         # print('相位%s，现在是路口%s，我是总延误时间哇%s' % (a+1,h, ddd))
         return round(ddd,4)


    def Doaction(self, t,a,demand):
        d=demand
        action = Action[a]
        totalnumber_onetime=0
        # step1:获取路口h各进口道的等待车辆数
        if t > 0:
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = self.n_m[i][t - 1][h] + self.i_mc[i][t - 1][h] - self.o_m[i][t - 1][h]  # 四个进口的车辆数
                    if self.n_m[i][t][h] <= 0:
                        self.n_m[i][t][h]=0
                    # print('self.n_m[i][t][h]', self.n_m[i][t][h])
        else:
            pass
        # for h in range(4):
        #     for i in self.n_m:
        # print('当前时刻为%s,road num is %s,进口道编号是 %s,等待车辆数是%s' %(t,h,i,self.n_m[i][t][h]))
        delay_fourroad=[]
        delay_total=0
        for h in range(4):
            # step2:计算当前时刻各进口道时延
            self.vc['区域1'][0]=7
            self.vc['区域2'][0]=8
            self.td_c['出口1'][t][h] = math.ceil(((self.l1[0] * 1000 * 3 - self.n_m['出口1'][t][h] * self.lv) / (3 * self.vc['区域1'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
            self.td_c['出口3'][t][h] = math.ceil(((self.l1[3] * 1000 * 3 - self.n_m['出口3'][t][h] * self.lv) / (3 * self.vc['区域2'][0] * self.C)))  # 为什么这里的社会车速度不是等于实时的区域平均速度
            if h == 0:
                self.td_c['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
            else:
                self.td_c['出口2'][t][h] =math.ceil(((self.lh[h - 1] * 1000 * 3 - self.n_m['出口2'][t][h] * self.lv) / (3 * self.vc['区域1'][0] * self.C)))
            if h == 3:
                self.td_c['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
            else:
                self.td_c['出口4'][t][h] =math.ceil(((self.lh[h] * 1000 * 3 - self.n_m['出口4'][t][h] * self.lv) / (3 * self.vc['区域2'][0] * self.C)))
            cc1 = int(t - abs(self.td_c['出口1'][t][h]))
            cc2 = int(t - abs(self.td_c['出口3'][t][h]))
            cc3 = int(t - abs(self.td_c['出口2'][t][h]))
            cc4 = int(t - abs(self.td_c['出口4'][t][h]))
            # step3:获得当前时刻各进口道增加的车辆需求
            ### 四个交叉口中，进口1和进口3的上游不存在交叉口，所以进入该交叉口的车辆数通过需求获得，进口2和进口4可能存在交叉口也可能不存在交叉口，所以进行分类计算
            if cc1 >= 0 and cc2 >= 0 and cc3 >= 0 and cc4 >= 0:
                self.i_mc['出口1'][t][h] = math.ceil(self.C * (0.25 * d[1_2][cc1]) +random.uniform(5, 20))# 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
                self.i_mc['出口3'][t][h] = math.ceil(self.C * (0.25 * d[2_1][cc2]) +random.uniform(5, 20))
                if h == 0:
                    self.i_mc['出口2'][t][h] = math.ceil(self.C * self.qc1  +random.uniform(0, 10)) # 非控制区域3的需求，直接通过到达率计算
                else:
                    self.i_mc['出口2'][t][h] = math.ceil(self.o_m['出口1'][cc3][h - 1] * self.alpha[2] + self.o_m['出口2'][cc3][h - 1] * \
                                                       self.alpha[1] + self.o_m['出口3'][cc3][h - 1] * self.alpha[0] +random.uniform(0, 5) ) # 存疑 ，
                if h == 3:
                    self.i_mc['出口4'][t][h] = math.ceil(self.C * self.qc2  +random.uniform(0, 10)) # 非控制区域4的需求，直接通过到达率计算
                else:
                    self.i_mc['出口4'][t][h] = math.ceil(self.o_m['出口1'][cc4][h + 1] * self.alpha[0] + self.o_m['出口3'][cc4][h + 1] * \
                                                       self.alpha[2] + self.o_m['出口4'][cc4][h + 1] * self.alpha[1] +random.uniform(0, 5))
            else:
                self.i_mc['出口1'][t][h] =10
                self.i_mc['出口3'][t][h] = 10
                if h == 0:
                    self.i_mc['出口2'][t][h] =10
                else:
                    self.i_mc['出口2'][t][h] =10
                if h == 3:
                    self.i_mc['出口4'][t][h] = 10
                else:
                    self.i_mc['出口4'][t][h] =10

            #step4:各进口道的左直右车辆数


            self.l_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[0])
            self.l_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[0])
            self.l_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[0])
            self.l_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[0])
            self.s_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[1])
            self.s_m['出口2'][t][h] =math.ceil( self.n_m['出口2'][t][h] * self.alpha[1])
            self.s_m['出口3'][t][h] = math.ceil(self.n_m['出口3'][t][h] * self.alpha[1])
            self.s_m['出口4'][t][h] = math.ceil(self.n_m['出口4'][t][h] * self.alpha[1])
            self.r_m['出口1'][t][h] = math.ceil(self.n_m['出口1'][t][h] * self.alpha[2])
            self.r_m['出口2'][t][h] = math.ceil(self.n_m['出口2'][t][h] * self.alpha[2])
            self.r_m['出口3'][t][h] =math.ceil( self.n_m['出口3'][t][h] * self.alpha[2])
            self.r_m['出口4'][t][h] =math.ceil( self.n_m['出口4'][t][h] * self.alpha[2])

            #step5：相位绿灯时间转移
            #相位1
            # print('进口4等待左转车辆数是%s,进口1等待直行的车辆数是%s,此次最大通过数为%s'% (self.l_m['出口4'][t][h],self.s_m['出口1'][t][h],self.sta_flow*action[0]))
            if self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]<self.sta_flow*action[0]:
                self.trans_l_m['出口4'][t][h] = self.l_m['出口4'][t][h]
                self.trans_s_m['出口1'][t][h] = self.s_m['出口1'][t][h]
                self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]

            else:
                self.trans_l_m['出口4'][t][h] = (self.l_m['出口4'][t][h]/(self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]))*self.sta_flow*action[0]
                self.trans_s_m['出口1'][t][h] = (self.l_m['出口1'][t][h]/(self.l_m['出口4'][t][h]+self.s_m['出口1'][t][h]))*self.sta_flow*action[0]
                self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
            zuozhuan1= self.l_m['出口4'][t][h].copy()# print('#####左转车辆数',zuozhuan1)         # print('#####直行车辆数',zhixing1)
            zhixing1=self.s_m['出口1'][t][h].copy()
            a=0
            ddd1=env.delayy(self, h, action, a, zuozhuan1, zhixing1)#a表示哪个相位，范围为0-3，h表示哪个路口
            # print('当前相位1的总延误时间',ddd1)
            #相位2
            if self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]<self.sta_flow*action[1]:
                self.trans_l_m['出口2'][t][h] = self.l_m['出口2'][t][h]
                self.trans_s_m['出口3'][t][h] = self.s_m['出口3'][t][h]
                self.trans_r_m['出口4'][t][h] = self.r_m['出口4'][t][h]
            else:
                self.trans_l_m['出口2'][t][h] = (self.l_m['出口2'][t][h]/(self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]))*self.sta_flow*action[1]
                self.trans_s_m['出口3'][t][h] = (self.l_m['出口3'][t][h]/(self.l_m['出口2'][t][h]+self.s_m['出口3'][t][h]))*self.sta_flow*action[1]
                self.trans_r_m['出口2'][t][h] = self.r_m['出口2'][t][h]
            zuozhuan2 = self.l_m['出口2'][t][h].copy()
            # print('#####左转车辆数', zuozhuan2)
            zhixing2 = self.s_m['出口3'][t][h].copy()
            # print('#####直行车辆数', zhixing2)
            a = 1
            ddd2 = env.delayy(self, h, action, a, zuozhuan2, zhixing2)  # a表示哪个相位，范围为0-3，h表示哪个路口
            # print('当前相位2的总延误时间', ddd2)
            # 相位3
            if self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h] < self.sta_flow * action[2]:
                self.trans_l_m['出口1'][t][h] = self.l_m['出口1'][t][h]
                self.trans_s_m['出口2'][t][h] = self.s_m['出口2'][t][h]
                self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
            else:
                self.trans_l_m['出口1'][t][h] = (self.l_m['出口1'][t][h]/(self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
                self.trans_s_m['出口2'][t][h] = (self.s_m['出口2'][t][h]/(self.l_m['出口1'][t][h] + self.s_m['出口2'][t][h])) * self.sta_flow * action[2]
                self.trans_r_m['出口3'][t][h] = self.r_m['出口3'][t][h]
            zuozhuan3 = self.l_m['出口1'][t][h].copy()
            # print('#####左转车辆数', zuozhuan3)
            zhixing3 = self.s_m['出口2'][t][h].copy()
            # print('#####直行车辆数', zhixing3)
            a = 2
            ddd3 = env.delayy(self, h, action, a, zuozhuan3, zhixing3)  # a表示哪个相位，范围为0-3，h表示哪个路口
            # print('当前相位3的总延误时间', ddd3)
            # 相位4
            if self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h] < self.sta_flow * action[3]:
                self.trans_l_m['出口3'][t][h] = self.l_m['出口3'][t][h]
                self.trans_s_m['出口4'][t][h] = self.s_m['出口4'][t][h]
                self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]
            else:
                self.trans_l_m['出口3'][t][h] = (self.l_m['出口3'][t][h] / (self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h]) )* self.sta_flow * action[3]
                self.trans_s_m['出口4'][t][h] = (self.s_m['出口4'][t][h]/ (self.l_m['出口3'][t][h] + self.s_m['出口4'][t][h])) * self.sta_flow * action[3]
                self.trans_r_m['出口1'][t][h] = self.r_m['出口1'][t][h]
            zuozhuan4 = self.l_m['出口3'][t][h].copy()
            # print('#####左转车辆数', zuozhuan4)
            zhixing4 = self.s_m['出口4'][t][h].copy()
            # print('#####直行车辆数', zhixing4)
            a = 3
            ddd4 = env.delayy(self, h, action, a, zuozhuan4, zhixing4)  # a表示哪个相位，范围为0-3，h表示哪个路口
            # print('当前相位4的总延误时间', ddd4)
            dd=ddd1+ddd2+ddd3+ddd4
            delay_total+=dd
            totalnumber_oneroad= self.l_m['出口1'][t][h]+self.l_m['出口2'][t][h]+self.l_m['出口3'][t][h]+self.l_m['出口4'][t][h]+\
                                 self.s_m['出口1'][t][h]+ self.s_m['出口2'][t][h]+ self.s_m['出口3'][t][h]+ self.s_m['出口4'][t][h]
            # totalnumber_oneroad=self.n_m['出口1'][t][h]+self.n_m['出口2'][t][h]+self.n_m['出口3'][t][h]+self.n_m['出口4'][t][h]
            totalnumber_onetime+=totalnumber_oneroad
            delay_fourroad.append(dd)
            # print('当前控制时间为%s,路口编号为%s,此控制区间的总延误为%s'%(t,h,dd))

            self.remain_s_m['出口1'][t][h] = self.s_m['出口1'][t][h] - self.trans_s_m['出口1'][t][h]
            self.remain_l_m['出口1'][t][h] = self.l_m['出口1'][t][h] - self.trans_l_m['出口1'][t][h]
            self.remain_r_m['出口1'][t][h] = 0

            self.remain_s_m['出口2'][t][h] = self.s_m['出口2'][t][h] - self.trans_s_m['出口2'][t][h]
            self.remain_l_m['出口2'][t][h] = self.l_m['出口2'][t][h] - self.trans_l_m['出口2'][t][h]
            self.remain_r_m['出口2'][t][h] = 0

            self.remain_s_m['出口3'][t][h] = self.s_m['出口3'][t][h] - self.trans_s_m['出口3'][t][h]
            self.remain_l_m['出口3'][t][h] = self.l_m['出口3'][t][h] - self.trans_l_m['出口3'][t][h]
            self.remain_r_m['出口3'][t][h] = 0

            self.remain_s_m['出口4'][t][h] = self.s_m['出口4'][t][h] - self.trans_s_m['出口4'][t][h]
            self.remain_l_m['出口4'][t][h] = self.l_m['出口4'][t][h] - self.trans_l_m['出口4'][t][h]
            self.remain_r_m['出口4'][t][h] = 0

            self.o_m['出口1'][t][h] = self.trans_s_m['出口1'][t][h]+self.trans_r_m['出口1'][t][h]+self.trans_l_m['出口1'][t][h]
            self.o_m['出口2'][t][h] = self.trans_s_m['出口2'][t][h]+self.trans_r_m['出口2'][t][h]+self.trans_l_m['出口2'][t][h]
            self.o_m['出口3'][t][h] = self.trans_s_m['出口3'][t][h]+self.trans_r_m['出口3'][t][h]+self.trans_l_m['出口3'][t][h]
            self.o_m['出口4'][t][h] = self.trans_s_m['出口4'][t][h]+self.trans_r_m['出口4'][t][h]+self.trans_l_m['出口4'][t][h]

            self.remain_m['出口1'][t][h] = self.remain_s_m['出口1'][t][h] + self.remain_r_m['出口1'][t][h] + self.remain_l_m['出口1'][t][h]
            self.remain_m['出口2'][t][h] = self.remain_s_m['出口2'][t][h] + self.remain_r_m['出口2'][t][h] + self.remain_l_m['出口2'][t][h]
            self.remain_m['出口3'][t][h] = self.remain_s_m['出口3'][t][h] + self.remain_r_m['出口3'][t][h] +  self.remain_l_m['出口3'][t][h]
            self.remain_m['出口4'][t][h] = self.remain_s_m['出口4'][t][h] + self.remain_r_m['出口4'][t][h] + self.remain_l_m['出口4'][t][h]



            self.actal_o_m['1-2'][t][h]=self.trans_l_m['出口4'][t][h]+self.trans_s_m['出口1'][t][h]+self.trans_r_m['出口2'][t][h]
            self.actal_o_m['2-1'][t][h] = self.trans_l_m['出口2'][t][h] + self.trans_s_m['出口3'][t][h] + self.trans_r_m['出口4'][t][h]
            # print('1-2转移流',t,h,self.actal_o_m['1-2'][t][h])

        AV_delay=delay_total/totalnumber_onetime
        # print('当前时刻四个路口总延误时间%s,当前时刻四个路口总的交通流%s,每个路口的总延误时间%s,车均延误为%s'%(delay_total,totalnumber_onetime,delay_fourroad,AV_delay))
        a12=self.actal_o_m['1-2'][t][0]+self.actal_o_m['1-2'][t][1]+self.actal_o_m['1-2'][t][2]+self.actal_o_m['1-2'][t][3]
        self.ZhuanYi1to2.append(a12)
        b21= self.actal_o_m['2-1'][t][0] + self.actal_o_m['2-1'][t][1] + self.actal_o_m['2-1'][t][2] + self.actal_o_m['2-1'][t][3]

        self.ZhuanYi2to1.append(b21)
        # print('1-2转移流', t, h, self.ZhuanYi1to2)
        # print('2-1转移流', t, h, self.ZhuanYi2to1)
        reward = -(abs(M1_2[t] - a12) + abs(M2_1[t] - b21))
        # print('区域1——2预期转移值is%s,区域1——2实际转移值is%s,差值是 %s' % (round(M1_2[t], 2),round(a12, 2),round(abs(M1_2[t] - a12), 2)))
        # print('区域2——1预期转移值is%s,区域2——1实际转移值is%s,差值是 %s' % (round(M2_1[t], 2), round(b21, 2), round(abs(M2_1[t] - b21), 2)))
        # print('两个区域的转移差值是%s'%(-reward))
        state = []
        for h in range(4):
            # print(h)
            for i in self.n_m:
                # print(i)
                self.n_m[i][t+1][h] = self.n_m[i][t][h] + self.i_mc[i][t][h] - self.o_m[i][t][h]
                state.append(self.n_m[i][t][h])
                # print(state)
                # print(len(state))
        state.append(M1_2[t+1])
        state.append(M2_1[t+1])
        # print(reward)
        if t==control_step:
            done=True
        else:
            done = False
        return  reward,state,done,delay_total,totalnumber_onetime
env=Env(object)

class QActorCriticAgent:
    def __init__(self, Env):
        self.gamma = 0.99

        self.actor_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ],
            output_size=env.action_space.n, output_activator=nn.Softmax(1))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)

        self.critic_net = self.build_net(
            input_size=env.observation_space.shape[0],
            hidden_sizes=[100, ],
            output_size=env.action_space.n)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.002)
        self.critic_loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip(
                [input_size, ] + hidden_sizes, hidden_sizes + [output_size, ]):
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ReLU())
        layers = layers[:-1]
        if output_activator:
            layers.append(output_activator)
        net = nn.Sequential(*layers)
        return net

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []
            self.discount = 1.

    def step(self, observation, reward, done):
        ##根据状态值选择动作
        state_tensor = torch.as_tensor(observation, dtype=torch.float).reshape(1, -1)
        prob_tensor = self.actor_net(state_tensor)
        # print('prob_tensor',prob_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        # print('action_tensor', action_tensor)
        action = action_tensor.numpy()[0] # #tensor转numpy数组
        # print('action', action)

        if self.mode == 'train':
            self.trajectory += [observation, reward, done, action]
            # print(self.trajectory)

            if len(self.trajectory) >= 8:
                self.learn()
            self.discount *= self.gamma
        return action

    def close(self):
        pass

    def learn(self):
        state, _, _, action, next_state, reward, done, next_action \
            = self.trajectory[-8:]
        state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float).unsqueeze(0)

        # train actor
        q_tensor = self.critic_net(state_tensor)[0, action]
        pi_tensor = self.actor_net(state_tensor)[0, action]
        logpi_tensor = torch.log(pi_tensor.clamp(1e-6, 1.))
        actor_loss_tensor = -self.discount * q_tensor * logpi_tensor
        self.actor_optimizer.zero_grad()
        actor_loss_tensor.backward()
        self.actor_optimizer.step()

        # train critic
        next_q_tensor = self.critic_net(next_state_tensor)[:, next_action]
        target_tensor = reward + (1. - done) * self.gamma * next_q_tensor
        pred_tensor = self.critic_net(state_tensor)[:, action]
        critic_loss_tensor = self.critic_loss(pred_tensor, target_tensor)
        self.critic_optimizer.zero_grad()
        critic_loss_tensor.backward()
        self.critic_optimizer.step()

agent = QActorCriticAgent(env)
def play_episode(env, agent, max_episode_steps=None, mode=None, render=False):
    Episode_reward = []
    MeanRewardPerStep = []
    time = []
    delay_totaltime = 0
    totalnumber_totaltime = 0

    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        action = agent.step(observation, reward, done)##选择合适的动作
        if done:
            break
        observation, reward, done, _ = env.step(action)##执行动作
        # r, s_, done, delay_total, totalnumber_onetime = env.Doaction(huanjing, t, a, demand)
        episode_reward += reward
        elapsed_steps += 1
        if max_episode_steps and elapsed_steps >= max_episode_steps:
            # max_episode_steps=500
            break
    agent.close()
    return episode_reward, elapsed_steps

logging.info('==== train ====')
episode_rewards = []
for episode in range(1):
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<OriginEpisode: %s' % episode)
    t = 0  # 重置环境
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    episode_reward, elapsed_steps = play_episode(env, agent,
            max_episode_steps=env._max_episode_steps, mode='train')
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-10:]) > -120:
        break
plt.plot(episode_rewards)










class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
        # print(x.shape)
        x = F.relu(self.fc1(x))                                                 # 连接输入层到隐藏层，且使用激励函数ReLU来处理经过隐藏层后的值
        actions_value = self.out(x)                                             # 连接隐藏层到输出层，获得最终的输出值 (即动作值)
        return actions_value                                                    # 返回动作值

# 定义DQN类 (定义两个网络)
class DQN(object):
    def __init__(self):                                                         # 定义DQN的一系列属性
        self.eval_net, self.target_net = Net(), Net()                           # 利用Net创建两个神经网络: 评估网络和目标网络
        self.learn_step_counter = 0                                             # for target updating
        self.memory_counter = 0                                                 # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))             # 初始化记忆库，一行代表一个transition
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)    # 使用Adam优化器 (输入为评估网络的参数和学习率)
        self.loss_func = nn.MSELoss()                                           # 使用均方损失函数 (loss(xi, yi)=(xi-yi)^2)

    def choose_action(self, x):                                                 # 定义动作选择函数 (x为状态)
        x = torch.unsqueeze(torch.FloatTensor(x), 0)                            # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
        if np.random.uniform() < EPSILON:                                       # 生成一个在[0, 1)内的随机数，如果小于EPSILON，选择最优动作
            actions_value = self.eval_net.forward(x)                            # 通过对评估网络输入状态x，前向传播获得动作值
            action = torch.max(actions_value, 1)[1].data.numpy()                # 输出每一行最大值的索引，并转化为numpy ndarray形式
#           values,indices = torch.max(input, dim)
            action_index = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action_index = np.random.randint(0, 9)                            # 这里action随机等于0或1 (N_ACTIONS = 2)

        return action_index                                                           # 返回选择的动作 (0或1)

    def Testchoose_action(self, x):  # 定义动作选择函数 (x为状态)
         x = torch.unsqueeze(torch.FloatTensor(x), 0)  # 将x转换成32-bit floating point形式，并在dim=0增加维数为1的维度
         actions_value = self.eval_net.forward(x)  # 通过对评估网络输入状态x，前向传播获得动作值
         action = torch.max(actions_value, 1)[1].data.numpy()  # 输出每一行最大值的索引，并转化为numpy ndarray形式
        #           values,indices = torch.max(input, dim)
         action_index = action[0]  # 输出action的第一个数
         return action_index  # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
        # print('self.memory.shape',self.memory.shape)
        # print('transition', transition)
        index = self.memory_counter % MEMORY_CAPACITY                           # 获取transition要置入的行数
        self.memory[index, :] = transition                                      # 置入transition
        self.memory_counter += 1                                                # memory_counter自加1

    def learn(self):  # 定义学习函数(记忆库已满后便开始学习)
        # 目标网络参数更新
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:  # 一开始触发，然后每100步触发
            self.target_net.load_state_dict(self.eval_net.state_dict())  # 将评估网络的参数赋给目标网络
        self.learn_step_counter += 1  # 学习步数自加1

        # 抽取记忆库中的批数据
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # 在[0, 2000)内随机抽取32个数，可能会重复
        b_memory = self.memory[sample_index, :]  # 抽取32个索引对应的32个transition，存入b_memory
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        # 将32个s抽出，转为32-bit floating point形式，并存储到b_s中，b_s为32行4列
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1].astype(int))
        # 将32个a抽出，转为64-bit integer (signed)形式，并存储到b_a中 (之所以为LongTensor类型，是为了方便后面torch.gather的使用)，b_a为32行1列
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        # 将32个r抽出，转为32-bit floating point形式，并存储到b_s中，b_r为32行1列
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])
        # 将32个s_抽出，转为32-bit floating point形式，并存储到b_s中，b_s_为32行4列

        # 获取32个transition的评估值和目标值，并利用损失函数和优化器进行评估网络参数更新
        q_eval = self.eval_net(b_s).gather(1, b_a)
        # eval_net(b_s)通过评估网络输出32行每个b_s对应的一系列动作值，然后.gather(1, b_a)代表对每行对应索引b_a的Q值提取进行聚合
        q_next = self.target_net(b_s_).detach()
        # q_next不进行反向传递误差，所以detach；q_next表示通过目标网络输出32行每个b_s_对应的一系列动作值
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        # q_next.max(1)[0]表示只返回每一行的最大值，不返回索引(长度为32的一维张量)；.view()表示把前面所得到的一维张量变成(BATCH_SIZE, 1)的形状；最终通过公式得到目标值
        loss = self.loss_func(q_eval, q_target)
        # 输入32个评估值和32个目标值，使用均方损失函数
        self.optimizer.zero_grad()                                      # 清空上一步的残余更新参数值
        loss.backward()                                                 # 误差反向传播, 计算参数更新值
        self.optimizer.step()                                           # 更新评估网络的所有参数
dqn = DQN()




