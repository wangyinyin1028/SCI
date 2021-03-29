import random
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from PycharmProjects.MPC.est_de import TrafficDemand
from PycharmProjects.MPC.est_OrigMPC import MPC_origin

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
f5 = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\aa.pkl', 'rb')
bb = pickle.load(f5)
f5.close()
d=bb.d_2
d_low = bb.d_low_2
# #############产生高需求与低需求，两个场景###################
demand = d  # 需求模式选择
U21 = [1]
U12 = [1]
x = [1]
M1_2 = [20]
M2_1 = [20]
step=100#控制方案

class MPC():
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
            self.t_c = 240  # 控制时间 单位为秒
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
            self.M1_2 = self.xx[0]
            self.M2_1 = self.xx[1]
            self.U21 = self.xx[2]
            self.U12 = self.xx[3]
            self.x = self.xx[4]
            self.N1 = self.xx[5]
            self.N2 = self.xx[6]
            self.m1 = self.xx[7]
            self.m2 = self.xx[8]


    def canshu(self):

        return self.n_p, self.N_p, self.G, self.m, self.vc, self.N1_C, self.N2_C, self.N1_JAM, self.N2_JAM, self.l, self.l1, self.l2, self.t_p, self.t_c, self.C, self.t_s, self.pc, self.pm, self.pop_size, self.u_num, self.t_s, self.pop_size, self.u_num

    # 1.设置U值属于0-1,一个u的基因编码为7个数，一个个体包含40个u，前10个是u12，后10个u21
    def jingdu(self,p, b):
        fanwei = b - p
        fenshu = 1000
        for i in range(50):
            # print(2**i)
            w = 2 ** i
            if w > (fanwei) * (fenshu):
                # i=10
                return i
            else:
                chrom_length = i

    def pop1(self,pop_size, u_num, chrom_length):
        pop = []
        for i in range(pop_size):
            geti = []
            for j in range(u_num):
                tem = []
                for k in range(chrom_length):
                    s = random.randint(0, 1)
                    tem.append(s)
                # print(tem)
                geti.append(tem)
            # print(geti)
            pop.append(geti)
        # print(pop)
        return pop

    def jiema(self,pop, pop_size, u_num, chrom_length):
        # pop=pop(getishuliang,u_num,jiyinchangdu)
        up = []
        for i in range(pop_size):
            temp = []
            for j in range(u_num):
                t = 0
                for k in range(chrom_length):
                    t += pop[i][j][k] * (math.pow(2, k))
                # print(t)
                u = t / (math.pow(2, chrom_length) - 1)
                temp.append(u)
                t = 0
            up.append(temp)
        return (up)

    ################################################
    # 3.根据守恒方程把u值放入，评价每个个体水平
    def PingJiaGeTi(self,pop, chrom_length, curent_t, demand):  ### 通过MPC获得预测步数内的区域内累计车辆数和完成流，以m/N的最大值作为适应度
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        pingjiazhi = []
        d = demand
        u = self.jiema(pop, pop_size, u_num, chrom_length)  # 40个个体，每个个体含有20个值，前10个为U21，后10个为U12
        # print('len(u)',len(u))
        for i in range(len(u)):
            # print(u[i])
            f1 = 0
            f2 = 0
            for k in range(curent_t, curent_t + t_p):
                n_p[1_1][k + 1] = n_p[1_1][k] + C * (d[1_1][k]) + t_c * (u[i][k - curent_t + 5] * m[2_1][k] - m[1_1][
                    k])  # 调用decodechrom(pop,chrom_length)后u的前20个值是1—2，后20个值是2—1
                # print('n_p[1_1][k + 1]',n_p[1_1][k + 1])
                n_p[1_2][k + 1] = n_p[1_2][k] + C * (d[1_2][k]) - t_c * (u[i][k - curent_t] * m[1_2][k])
                n_p[2_2][k + 1] = n_p[2_2][k] + C * (d[2_2][k]) + t_c * (u[i][k - curent_t] * m[1_2][k] - m[2_2][k])
                n_p[2_1][k + 1] = n_p[2_1][k] + C * (d[2_1][k]) - t_c * (u[i][k - curent_t + 5] * m[2_1][k])
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

                f1 += (N_p['区域1'][k + 1] - N1_C) ** 2
                # + (N_p['区域2'][k + 1]-N2_C)**2
                # f2 += m[2_2][k + 1] + m[1_1][k + 1]  # 由于需求设置使得G为0导致m为0，故f2为0，存在问题（可能是G的函数存在问题）

            f = 1 / f1
            pingjiazhi.append(f)
        # print(obj_value)
        # print(len(obj_value))

        return pingjiazhi

    def calfiValue(self,pingjiazhi):
        fit_value = []
        # c_min = 0
        # (m, obj_value) = calobjValue(pop, chrom_length)

        for i in range(len(pingjiazhi)):
            if (pingjiazhi[i] > 0 and pingjiazhi[i] < float('inf')):
                temp = pingjiazhi[i]
            else:
                temp = 0.0
            fit_value.append(temp)
        # print(fit_value)
        return fit_value

    def sum(self,fit_value):
        total = 0
        for i in range(len(fit_value)):
            total += fit_value[i]
        return total

    def cumsum(self,newfit):
        for i in range(len(newfit) - 2, -1, -1):
            t = 0
            j = 0
            while (j <= i):
                t += newfit[j]
                j += 1
            newfit[i] = t
            newfit[len(newfit) - 1] = 1

    def selection(self,pop, fit_value):
        ms = []
        for i in range(len(pop)):
            ms.append(random.random())
        ms.sort()
        # print(ms)

        newfit = []
        totalfit = sum(fit_value)
        for i in range(len(fit_value)):
            newfit.append(fit_value[i] / totalfit)
        # print(newfit)
        self.cumsum(newfit)

        fitin = 0
        newin = 0
        newpop = pop
        while newin < len(pop) and fitin < len(pop):
            if (ms[newin] < newfit[fitin]):
                newpop[newin] = pop[fitin]
                newin += 1
            else:
                fitin += 1
        pop = newpop
        return pop

    def crossover(self,pop, pc):
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        for i in range(len(pop) - 1):
            for j in range(u_num):
                if (random.random() < pc):
                    cpoint = random.randint(0, self.jingdu(0, 1))
                    temp1 = []
                    temp2 = []
                    temp1.extend(pop[i][j][0:cpoint])
                    temp1.extend(pop[i + 1][j][cpoint:len(pop[i][j])])
                    temp2.extend(pop[i + 1][j][0:cpoint])
                    temp2.extend(pop[i + 1][j][cpoint:len(pop[i][j])])
                    pop[i][j] = temp1
                    pop[i + 1][j] = temp2
        return pop

    def mutation(self,pop, pm):
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        for i in range(len(pop)):
            for j in range(u_num):
                if (random.random() < pm):
                    mpoint = random.randint(0, self.jingdu(0, 1) - 1)
                    if (pop[i][j][mpoint] == 1):
                        pop[i][j][mpoint] = 0
                    else:
                        pop[i][j][mpoint] = 1
        return pop
        # 找出最优值

    def best(self,pop, fit_value):
        px = len(pop)
        best_individual = []
        best_fit = fit_value[0]
        for i in range(px):
            if fit_value[i] > best_fit:
                # print(best_fit)
                # print(fit_value[i])
                best_fit = fit_value[i]
                best_individual = pop[i]
            else:
                best_individual = pop[0]

        # print(best_fit)

        return best_individual, best_fit

    # 将二进制转化为十进制
    def b2d(self,best_individual):
        chrom_length = self.jingdu(0, 1)
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        length = len(best_individual)
        u_best = []
        t = 0
        for i in range(int(2 * t_p)):
            for k in range(self.jingdu(0, 1)):
                t += best_individual[i][k] * (math.pow(2, k))
            u = t / (math.pow(2, chrom_length) - 1)
            u_best.append(u)
            t = 0
        a = [u_best[0], u_best[5]]
        return a  # u_best[0]是在t时刻u12的最优值，u_best[19]是在t时刻u21的最优值

    def ControlMpc(self,demand,step):
        n_p, N_p, G, m, vc, N1_C, N2_C, N1_JAM, N2_JAM, l, l1, l2, t_p, t_c, C, t_s, pc, pm, pop_size, u_num, t_s, pop_size, u_num = self.canshu()
        chrom_length = self.jingdu(0, 1)
        s = 1
        N1=[]
        N2=[]
        m1=[]
        m2=[]

        for j in range(step):
            # print('est_mpc文件################now is ',j)
            if_stop = False  # 是否停止循环，差值小于0.5时停止
            while not if_stop:
                curent_t = j
                pop = self.pop1(pop_size, u_num, chrom_length)  # 产生初始种群
                for i in range(30):
                    fit = self.PingJiaGeTi(pop, chrom_length, curent_t,
                                            demand)  # 从0时刻开始，计算40套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
                    fitvalue = self.calfiValue(fit)  # 40个适应值
                    pop = self.selection(pop, fitvalue)  # 30套方案，种群选择
                    pop = self.crossover(pop, pc)
                    pop = self.mutation(pop, pm)
                fit = self.PingJiaGeTi(pop, chrom_length, curent_t,
                                        demand)  # 从0时刻开始，计算30套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
                fitvalue = self.calfiValue(fit)  # 40个适应值
                best_individual, best_fit = self.best(pop, fitvalue)
                u2 = self.b2d(best_individual)
                for i in range(2):
                    if u2[i] <= 0.1:
                        u2[i] = 0.1
                for i in range(2):
                    if u2[i] > 1:
                        u2[i] = 1
                off_u21 = abs(u2[0] - U21[-1])
                off_u12 = abs(u2[1] - U12[-1])
                # if off_u21 <= 0.3 and off_u12 <= 0.5 and int(n_p[1_1][curent_t] + n_p[1_2][curent_t]) < 4000 and int(n_p[1_1][curent_t] + n_p[1_2][curent_t]) < 2500:
                if  int(n_p[1_1][curent_t] + n_p[1_2][curent_t])<9000:
                    U21.append(u2[0])
                    U12.append(u2[1])
                    n1=int(n_p[1_1][curent_t] + n_p[1_2][curent_t])
                    n2=int(n_p[2_1][curent_t] + n_p[2_2][curent_t])
                    # print('区域1车辆数', n1)
                    # print('区域2车辆数', n2)
                    G1 = -1E-07 * pow(n1, 2) + 4E-04 * (n1) + 0.022
                    G2 = -2E-07 * pow(n2, 2) + 5E-04 * (n2) + 0.0478
                    if G1 < 0:
                        G1 = 0
                    if G2 < 0:
                        G2 = 0
                    if n1 == 0 or G1 == 0:
                        mm11 = 0
                        mm12 = 0
                    else:
                        mm11 = (n_p[1_1][curent_t] / n1) * G1
                        mm12 = (n_p[1_2][curent_t] / n1) * G1
                    if n2 == 0 or G2 == 0:
                        mm21 = 0
                        mm22 = 0
                    else:
                        mm21 = (n_p[2_1][curent_t] / n2) * G2
                        mm22 = (n_p[2_2][curent_t] / n2) * G2
                    m11=mm11+mm12
                    m22=mm22+mm21
                    # print('区域1转移车辆数', round(m11,4))
                    # print('区域2转移车辆数', round(m22,4))
                    N1.append(n1)
                    N2.append(n2)
                    m1.append(m11)
                    m2.append(m22)
                    com21 = (U21[0] * m[2_1][curent_t]) * t_c
                    com12 = (U12[1] * m[1_2][curent_t]) * t_c
                    M1_2.append(int(com21))
                    M2_1.append(int(com12))
                    s = s + 1
                    x.append(s)
                    if_stop = True
                else:
                    pass
        return x, U21, U12, M2_1, M1_2,N1,N2,m1,m2
    def Transform(self):
        WW=[]
        x, U21, U12, M2_1, M1_2,N1,N2,m1,m2 = self.ControlMpc(demand,step)
        WW.append(M2_1)
        WW.append(M1_2)
        WW.append(U21)
        WW.append(U12)
        WW.append(x)
        WW.append(N1)
        WW.append(N2)
        WW.append(m1)
        WW.append(m2)
        return WW

def picture_u(x, w1, w2):
        plt.figure(figsize=(15, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.xlabel('time(2min)')
        plt.ylabel('U')
        plt.step(x, w1, color="#FF6347", where="pre", lw=1, label='u21')
        plt.step(x, w2, color="#0000FF", where="pre", lw=1, label='u12')
        plt.legend(loc='upper left')
        plt.title('低需求边界控制')
        plt.grid(True)
        plt.show()
def picture_m(x, w1, w2):
    plt.figure(figsize=(15, 4))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('time(2min)')
    plt.ylabel('预期转移值')
    plt.step(x, w1, color="#FF6347", where="pre", lw=1, label='M21')
    plt.step(x, w2, color="#0000FF", where="pre", lw=1, label='M12')
    plt.legend(loc='upper left')
    plt.title('低需求下预期转移值')
    plt.grid(True)
    plt.show()
def picture_mfd1(N,M,n,m):
    plt.xlabel('累计车辆数(辆)')
    plt.ylabel('累计完成量(辆/秒)')
    colors1 = '#000000'  # 点的颜色
    colors2 = '#FF0000'  # 点的颜色
    area = np.pi * 2 ** 2  # 点面积
    area2 = np.pi * 2 ** 2  # 点面积
    plt.scatter(N, M, s=area, c=colors1, alpha=0.4, label='区域1受控制')
    plt.scatter(n, m, s=area2, c=colors2, alpha=0.4, label='区域1未控制')
    plt.title('高需求')
    plt.legend()
    plt.show()
def picture_mfd2(N,M,n,m):
    plt.xlabel('累计车辆数(辆)')
    plt.ylabel('累计完成量(辆/秒)')
    colors1 = '#000000'  # 点的颜色
    colors2 = '#FF0000'  # 点的颜色
    area1 = np.pi * 2 ** 2  # 点面积
    area2 = np.pi * 2 ** 2  # 点面积
    plt.scatter(N, M, s=area1, c=colors1, alpha=0.4, label='区域2受控制')
    plt.scatter(n, m, s=area2, c=colors2, alpha=0.4, label='区域2未控制')
    plt.title('高需求')
    plt.legend()
    plt.show()

aa = MPC()
f = open(r'E:\wyy\PycharmProjects\MPC\temp_file\bb.pkl', 'wb')
pickle.dump(aa, f, 0)
f.close()


# bbb=aa
# M1_2,M2_1,U21,U12,x,N1,N2,m1,m2=bbb.M1_2,bbb.M2_1,bbb.U21,bbb.U12,bbb.x,bbb.N1,bbb.N2,bbb.m1,bbb.m2
###########################################原始方案############################################################################################
# f3 = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\MPC_origin.pkl', 'rb')
# tt = pickle.load(f3)
# f3.close()
# n1_orig,n2_orig,m1_orig,m2_orig =tt.n1_orig,tt.n2_orig,tt.m1_orig,tt.m2_orig
# picture_mfd1(N1,m1,n1_orig,m1_orig)
# picture_mfd2(N2,m2,n2_orig,m2_orig)




























