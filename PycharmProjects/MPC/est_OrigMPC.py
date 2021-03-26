import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from PycharmProjects.MPC.est_de import TrafficDemand

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
f5 = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\aa.pkl', 'rb')
bb = pickle.load(f5)
f5.close()
d=bb.d_2
d_low = bb.d_low_2

# #############产生高需求与低需求，两个场景###################
demand = d  # 需求模式选择
##########################################################
U21 = [1]
U12 = [1]
x = [1]
M1_2 = [20]
M2_1 = [20]
# step=60#控制方案
step2=30#放入原始方案

class MPC_origin():
    def __init__(self):
            self.n_p = {1_1: np.zeros(900), 1_2: np.zeros(900), 2_1: np.zeros(900), 2_2: np.zeros(900)}  # 存放四状态
            self.N_p = {'区域1': np.zeros(900), '区域2': np.zeros(900)}  # 存放两区域
            # d已经产生
            self.G = {'区域1': np.zeros(900), '区域2': np.zeros(900)}  # 存放两区域
            self.m = {1_1: np.zeros(900), 1_2: np.zeros(900), 2_1: np.zeros(900), 2_2: np.zeros(900)}  # 存放实际转移流
            self.vc = {'区域1': np.ones(900), '区域2': np.ones(900)}
            self.N1_C = 2396
            self.N2_C = 1338
            self.N1_JAM = 4000
            self.N2_JAM = 2500
            self.l = [31.04, 21.415]
            self.l1 = [3.1, 2.9, 3.0, 3.3]
            self.l2 = [1.9, 2.1, 2.0, 1.3]
            self.t_p = 5  # 预测步数
            self.t_c = 180  # 控制时间 单位为秒
            self.C = 120
            self.t_s = 300  # 控制步数
            self.n_p[1_1][0] = 50
            self.n_p[1_2][0] = 50
            self.n_p[2_1][0] = 50
            self.n_p[2_2][0] = 40
            self.N_p['区域1'][0] = 100
            self.N_p['区域2'][0] = 90
            self.pc = 0.8
            self.pm = 0.05
            self.pop_size = 40  # 40个个体
            self.u_num = 10  # 每个个体有20个值
            self.t_s = 300  # 控制步数为300步
            self.pop_size = 40  # 40个个体
            self.u_num = 10  # 每个个体有20个值
            self.xx = self.Transform()
            self.n1_orig = self.xx[0]
            self.n2_orig = self.xx[1]
            self.m1_orig = self.xx[2]
            self.m2_orig= self.xx[3]
    def canshu(self):
        return self.n_p, self.N_p, self.G, self.m, self.vc, self.N1_C, self.N2_C, self.N1_JAM, self.N2_JAM, self.l, self.l1, self.l2, self.t_p, self.t_c, self.C, self.t_s, self.pc, self.pm, self.pop_size, self.u_num, self.t_s, self.pop_size, self.u_num
    def PingJia(self,demand,step2):  ### 通过MPC获得预测步数内的区域内累计车辆数和完成流，以m/N的最大值作为适应度
        n1_orig = []
        n2_orig = []
        m1_orig=[]
        m2_orig = []
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        pingjiazhi = []
        d = demand
        u = 1
        f1 = 0
        f2 = 0
        for k in range(step2):
                n_p[1_1][k + 1] = n_p[1_1][k] + C * (d[1_1][k]) + t_c * (u * m[2_1][k] - m[1_1][k])
                n_p[1_2][k + 1] = n_p[1_2][k] + C * (d[1_2][k]) - t_c * (u * m[1_2][k])
                n_p[2_2][k + 1] = n_p[2_2][k] + C * (d[2_2][k]) + t_c * (u * m[1_2][k] - m[2_2][k])
                n_p[2_1][k + 1] = n_p[2_1][k] + C * (d[2_1][k]) - t_c * (u * m[2_1][k])
                if n_p[1_1][k + 1] < 0:
                    n_p[1_1][k + 1] = 0
                elif n_p[1_2][k + 1] < 0:
                    n_p[1_2][k + 1] = 0
                elif n_p[2_2][k + 1] < 0:
                    n_p[2_2][k + 1] = 0
                elif n_p[2_1][k + 1] < 0:
                    n_p[2_1][k + 1] = 0

                N_p['区域1'][k + 1] = n_p[1_1][k + 1] + n_p[1_2][k + 1]
                N_p['区域2'][k + 1] = n_p[2_1][k + 1] + n_p[2_2][k + 1]
                # print('区域1车辆数', n_p[1_1][k + 1] + n_p[1_2][k + 1])
                # print('区域1车辆数', n_p[1_1][k + 1] + n_p[1_2][k + 1])

                # G['区域1'][k + 1] = -3E-09 * pow(N_p['区域1'][k + 1], 2) + 6E-05 * (N_p['区域1'][k + 1]) + 0.0183
                # G['区域2'][k + 1] = -6E-09 * pow(N_p['区域2'][k + 1], 2) + 8E-05 * (N_p['区域2'][k + 1]) + 0.0399
                G['区域1'][k + 1] = -1E-07 * pow(N_p['区域1'][k + 1], 2) + 4E-04 * (N_p['区域1'][k + 1]) + 0.022
                G['区域2'][k + 1] = -2E-07 * pow(N_p['区域2'][k + 1], 2) + 5E-04 * (N_p['区域2'][k + 1]) + 0.0478

                # print(vc)
                if G['区域1'][k + 1] < 0:
                    G['区域1'][k + 1] = 0
                if G['区域2'][k + 1] < 0:
                    G['区域2'][k + 1] = 0
                if N_p['区域1'][k + 1] == 0 or G['区域1'][k + 1] == 0:
                    m[1_1][k + 1] = 0
                    m[1_2][k + 1] = 0
                else:
                    m[1_1][k + 1] = (n_p[1_1][k + 1] / N_p['区域1'][k + 1]) * G['区域1'][k + 1]
                    m[1_2][k + 1] = (n_p[1_2][k + 1] / N_p['区域1'][k + 1]) * G['区域1'][k + 1]
                if N_p['区域2'][k + 1] == 0 or G['区域2'][k + 1] == 0:
                    m[2_1][k + 1] = 0
                    m[2_2][k + 1] = 0
                else:
                    m[2_1][k + 1] = (n_p[2_1][k + 1] / N_p['区域2'][k + 1]) * G['区域2'][k + 1]
                    m[2_2][k + 1] = (n_p[2_2][k + 1] / N_p['区域2'][k + 1]) * G['区域2'][k + 1]
                m1=m[1_1][k + 1]+m[1_2][k + 1]
                m2=m[2_1][k + 1]+m[2_2][k + 1]
                n1_orig.append(N_p['区域1'][k + 1])
                n2_orig.append(N_p['区域2'][k + 1])
                m1_orig.append(m1)
                m2_orig.append(m2)
                # print('##########################当前控制步数###########################', k)
                # print('区域1车辆数',round(N_p['区域1'][k + 1],4))
                # print('区域2车辆数',round(N_p['区域2'][k + 1],4))

                if N_p['区域1'][k + 1] > 0:
                    vc['区域1'][k + 1] = G['区域1'][k + 1] / (N_p['区域1'][k + 1] / (l[0] * 1000))
                else:
                    vc['区域1'][k + 1] = 16.7  # 限速60
                if N_p['区域1'][k + 1] > 0:
                    vc['区域2'][k + 1] = G['区域2'][k + 1] / (N_p['区域2'][k + 1] / (l[1] * 1000))
                else:
                    vc['区域2'][k + 1] = 16.7  # 限速60

                if vc['区域1'][k + 1] < 0:
                    vc['区域1'][k + 1] = 0
                # if vc['区域1'][k + 1] > 16.7:
                #     vc['区域1'][k + 1] = 16.7
                if vc['区域2'][k + 1] < 0:
                    vc['区域2'][k + 1] = 0
                # if vc['区域1'][k + 1] > 16.7:
                #     vc['区域1'][k + 1] = 16.7

        #         f1 += (N_p['区域1'][k + 1] - N1_C) ** 2
        #         # + (N_p['区域2'][k + 1]-N2_C)**2
        #         # f2 += m[2_2][k + 1] + m[1_1][k + 1]  # 由于需求设置使得G为0导致m为0，故f2为0，存在问题（可能是G的函数存在问题）
        #
        #     f = 1 / f1
        #     pingjiazhi.append(f)
        # # print(obj_value)
        # # print(len(obj_value))

        return n1_orig,n2_orig,m1_orig,m2_orig
    def Transform(self):
        WWW=[]
        n1_orig,n2_orig,m1_orig,m2_orig = self.PingJia(demand,step2)
        WWW.append(n1_orig)
        WWW.append(n2_orig)
        WWW.append(m1_orig)
        WWW.append(m2_orig)
        return WWW

def picture_mfd1(n,m):
    plt.xlabel('累计车辆数(辆)')
    plt.ylabel('累计完成量(辆/秒)')
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#000000'  # 点的颜色
    area = np.pi * 2 ** 2  # 点面积
    area2 = np.pi * 2 ** 2  # 点面积
    plt.scatter(n, m, s=area2, c=colors2, alpha=0.4, label='区域1未控制')
    plt.title('高需求')
    plt.legend()
    plt.show()
def picture_mfd2(n,m):
    plt.xlabel('累计车辆数(辆)')
    plt.ylabel('累计完成量(辆/秒)')
    colors1 = '#00CED1'  # 点的颜色
    colors2 = '#000000'  # 点的颜色
    area1 = np.pi * 2 ** 2  # 点面积
    area2 = np.pi * 2 ** 2  # 点面积
    plt.scatter(n, m, s=area2, c=colors2, alpha=0.4, label='区域2未控制')
    plt.title('高需求')
    plt.legend()
    plt.show()


e = MPC_origin()
f = open(r'E:\wyy\PycharmProjects\MPC\temp_file\MPC_origin.pkl', 'wb')
pickle.dump(e, f, 0)
f.close()
# tt=e
# n1_orig,n2_orig,m1_orig,m2_orig =tt.n1_orig,tt.n2_orig,tt.m1_orig,tt.m2_orig
# picture_mfd1(n1_orig,m1_orig)
# picture_mfd2(n2_orig,m2_orig)