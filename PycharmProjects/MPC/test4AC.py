import random
import sys
import logging
import importlib as imp
import matplotlib.pyplot as plt
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
Action={0:[10,50,30,30],1:[15,45,30,30],2:[20,40,30,30],3:[25,35,30,30],4:[30,30,30,30],5:[35,25,30,30],6:[40,20,30,30],7:[45,15,30,30],
        8:[50,10,25,35],9:[30,30,25,35],10:[30,30,20,40],11:[30,30,15,45],12:[30,30,10,50],13:[30,30,35,25],14:[30,30,40,20],15:[30,30,45,15],
        16:[30,30,50,10],17:[30,20,40,30]}
# AC算法
imp.reload(logging)
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s [%(levelname)s] %(message)s',stream=sys.stdout, datefmt='%H:%M:%S')
class Env():
    def __init__(self):
        self._max_episode_steps =99
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        for t in range(1):
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = 5
        self.td_c  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 私家车时延
        self.i_mc= {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车与公交车之和
        self.sta_flow  = 0.4  # 饱和流
        self.l_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.s_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.trans_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.trans_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.trans_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.remain_l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.remain_s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.remain_r_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.actal_o_m = {'1-2': np.zeros((900, 4)), '2-1': np.zeros((900, 4))}
        self.ZhuanYi1to2=[]
        self.ZhuanYi2to1 = []
        self.o_m  =  {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.remain_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),'出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.qc1 = 0.05
        self.qc2 = 0.05
        self.C = 120
        self.vc = {'区域1': np.ones(900), '区域2': np.ones(900)}

    def reset(self):
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
        # print('现在时间为%s,信号配时方案%s'%(t,action))
        if t > 0:
            for h in range(4):
                for i in self.n_m:
                    self.n_m[i][t][h] = self.n_m[i][t - 1][h] + self.i_mc[i][t - 1][h] - self.o_m[i][t - 1][h]  # 四个进口的车辆数
                    if self.n_m[i][t][h] <= 0:
                        self.n_m[i][t][h]=0
                    # print('self.n_m[i][t][h]', self.n_m[i][t][h])
        else:
            for t in range(1):
                for h in range(4):
                    for i in self.n_m:
                        self.n_m[i][t][h] = 5
        # for h in range(4):
            # for i in self.n_m:
                # print('时间为%s,当前时刻路口%s，进口道%s ，车辆数%s'%(t,h,i,self.n_m[i][t][h]))
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
            # zuozhuan1= self.l_m['出口4'][t][h].copy()# print('#####左转车辆数',zuozhuan1)         # print('#####直行车辆数',zhixing1)
            # zhixing1=self.s_m['出口1'][t][h].copy()
            # a=0
            # ddd1=env.delayy(self, h, action, a, zuozhuan1, zhixing1)#a表示哪个相位，范围为0-3，h表示哪个路口
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
            # zuozhuan2 = self.l_m['出口2'][t][h].copy()
            # # print('#####左转车辆数', zuozhuan2)
            # zhixing2 = self.s_m['出口3'][t][h].copy()
            # # print('#####直行车辆数', zhixing2)
            # a = 1
            # ddd2 = env.delayy(self, h, action, a, zuozhuan2, zhixing2)  # a表示哪个相位，范围为0-3，h表示哪个路口
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
            # zuozhuan3 = self.l_m['出口1'][t][h].copy()
            # # print('#####左转车辆数', zuozhuan3)
            # zhixing3 = self.s_m['出口2'][t][h].copy()
            # # print('#####直行车辆数', zhixing3)
            # a = 2
            # ddd3 = env.delayy(self, h, action, a, zuozhuan3, zhixing3)  # a表示哪个相位，范围为0-3，h表示哪个路口
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
            # zuozhuan4 = self.l_m['出口3'][t][h].copy()
            # # print('#####左转车辆数', zuozhuan4)
            # zhixing4 = self.s_m['出口4'][t][h].copy()
            # # print('#####直行车辆数', zhixing4)
            # a = 3
            # ddd4 = env.delayy(self, h, action, a, zuozhuan4, zhixing4)  # a表示哪个相位，范围为0-3，h表示哪个路口
            # # print('当前相位4的总延误时间', ddd4)
            # dd=ddd1+ddd2+ddd3+ddd4
            # delay_total+=dd
            # totalnumber_oneroad= self.l_m['出口1'][t][h]+self.l_m['出口2'][t][h]+self.l_m['出口3'][t][h]+self.l_m['出口4'][t][h]+\
            #                      self.s_m['出口1'][t][h]+ self.s_m['出口2'][t][h]+ self.s_m['出口3'][t][h]+ self.s_m['出口4'][t][h]
            # # totalnumber_oneroad=self.n_m['出口1'][t][h]+self.n_m['出口2'][t][h]+self.n_m['出口3'][t][h]+self.n_m['出口4'][t][h]
            # totalnumber_onetime+=totalnumber_oneroad
            # delay_fourroad.append(dd)
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

        # AV_delay=delay_total/totalnumber_onetime
        # print('当前时刻四个路口总延误时间%s,当前时刻四个路口总的交通流%s,每个路口的总延误时间%s,车均延误为%s'%(delay_total,totalnumber_onetime,delay_fourroad,AV_delay))
        a12=self.actal_o_m['1-2'][t][0]+self.actal_o_m['1-2'][t][1]+self.actal_o_m['1-2'][t][2]+self.actal_o_m['1-2'][t][3]
        self.ZhuanYi1to2.append(a12)
        b21= self.actal_o_m['2-1'][t][0] + self.actal_o_m['2-1'][t][1] + self.actal_o_m['2-1'][t][2] + self.actal_o_m['2-1'][t][3]
        # print('现在时间是%s,区域2转移至区域1车辆数%s'%(t,b21))
        # print('现在时间是%s,区域1转移至区域2车辆数%s' % (t, a12))
        # print('现在时间是%s,区域2转移至区域1车辆数预期转移值%s' % (t, M2_1[t]))
        # print('现在时间是%s,区域1转移至区域2车辆数预期转移值%s' % (t, M1_2[t]))

        self.ZhuanYi2to1.append(b21)
        # print('1-2转移流', t, h, self.ZhuanYi1to2)
        # print('2-1转移流', t, h, self.ZhuanYi2to1)
        reward = -(abs(M1_2[t] - a12) + abs(M2_1[t] - b21))
        # print('现在时间是%s,区域1转移至区域2车辆数转移值差距值%s' % (t, reward))
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
        if t==self._max_episode_steps:
            done=True
        else:
            done = False
        return  reward ,state,done
                # delay_total,totalnumber_onetime
env=Env()

class QActorCriticAgent:
    def __init__(self, env):
        self.gamma = 0.99

        self.actor_net = self.build_net(
            input_size=18,
            hidden_sizes=[100, ],
            output_size=18, output_activator=nn.Softmax(1))
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 0.001)

        self.critic_net = self.build_net(
            input_size=18,
            hidden_sizes=[100, ],
            output_size=18)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), 0.002)
        self.critic_loss = nn.MSELoss()

    def build_net(self, input_size, hidden_sizes, output_size=1, output_activator=None):
        layers = []
        for input_size, output_size in zip(
                [input_size, ] + hidden_sizes, hidden_sizes + [output_size, ]):
            layers.append(nn.Linear(int(input_size), int(output_size)))
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
        state_tensor = torch.as_tensor(state, dtype=torch.float).unsqueeze(0)#在第1维增加一个维度
        next_state_tensor = torch.as_tensor(next_state, dtype=torch.float).unsqueeze(0)

        # train actor
        q_tensor = self.critic_net(state_tensor)[0, action]
        pi_tensor = self.actor_net(state_tensor)[0, action]
        logpi_tensor = torch.log(pi_tensor.clamp(1e-6, 1.))#超过min和max部份截断，log以e为底的对数
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
    observation, reward, done = env.reset(), 0., False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    while True:
        a = agent.step(observation, reward, done)##选择合适的动作
        # print('action is ',a)
        if done:
            break
        reward, s_, done = env.Doaction(elapsed_steps, a, demand)
        episode_reward += reward
        elapsed_steps += 1
        if  elapsed_steps >= max_episode_steps:
            break
    agent.close()
    return episode_reward, elapsed_steps

logging.info('==== train ====')
episode_rewards = []
for episode in range(200):
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<OriginEpisode: %s' % episode)
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    episode_reward, elapsed_steps = play_episode(env, agent, max_episode_steps=env._max_episode_steps, mode='train')
    episode_rewards.append(episode_reward)
    logging.debug('train episode %d: reward = %.2f, steps = %d',
            episode, episode_reward, elapsed_steps)
    # if np.mean(episode_rewards[-10:]) > -120:
    #     break
def ResultTrain(Episode,Episode_reward):
    plt.figure(figsize=(12, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('训练次数(次）')
    plt.ylabel('奖赏值')
    plt.plot(Episode, Episode_reward, label='深度强化学习AC', linewidth=1)
    # plt.plot(x, y2, label='u12', linewidth=0.5)
    plt.legend(loc='upper left')
    plt.title('高需求情况下迭代结果')
    plt.grid(True)
    plt.show()
x=[]
for i in range(len(episode_rewards)):
    x.append(i)
ResultTrain(x,episode_rewards)















