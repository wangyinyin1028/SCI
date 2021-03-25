import numpy as np
import random
import math
import matplotlib.pyplot as plt
from demand import d

## MFD模型参数
N1 = 100
n11 = 80
n12 = 50
N2 = 160
n21 = 50
n22 = 40
N1_C = 2396
N2_C = 1338
N1_JAM = 5000
N2_JAM = 2500
l = [31.04, 21.415]
l1 =[3.1,2.9,3.0,3.3]
l2 =[1.9,2.1,2.0,1.3]
t_p = 5  # 预测步数
t_c = 300 # 控制时间 单位为秒
C = 100
t_s = 300  # 控制步数
G1 = -3E-08* pow(N1, 2) + 6E-04 *N1 + 0.0183
G2 = -6E-08*pow(N2, 2) + 8E-04 *N2  + 0.0399
m11 = (n11 / N1) * G1
m12 = (n12 / N1) * G1
m22 = (n22 / N2) * G2
m21 = (n21 / N1) * G2
#初始化

n_p={1_1:np.zeros(900),1_2:np.zeros(900),2_1:np.zeros(900),2_2:np.zeros(900)}#存放四状态
N_p={'区域1':np.zeros(900),'区域2':np.zeros(900)}#存放两区域
#d已经产生
G={'区域1':np.zeros(900),'区域2':np.zeros(900)}#存放两区域
m={1_1:np.zeros(900),1_2:np.zeros(900),2_1:np.zeros(900),2_2:np.zeros(900)}#存放实际转移流
vc = {'区域1': np.ones(900), '区域2': np.ones(900)}

#1.设置U值属于0-1,一个u的基因编码为7个数，一个个体包含40个u，前10个是u12，后10个u21

def jingdu(p,b):
    fanwei=b-p
    fenshu=1000
    for i in range(50):
        # print(2**i)
        w=2**i
        if w>(fanwei)*(fenshu):
           #i=10
            return i
        else:
            chrom_length=i
#2.设置种群大小即个体数量,有30个个体，每个个体由40个u组成，每个u由7个基因编码

def pop1(pop_size,u_num,chrom_length):
    pop=[]
    for i in range(pop_size):
        geti=[]
        for j in range(u_num):
            tem = []
            for k in range(chrom_length):
                s=random.randint(0,1)
                tem.append(s)
            # print(tem)
            geti.append(tem)
     # print(geti)
        pop.append(geti)
    # print(pop)
    return pop
def jiema(pop,pop_size,u_num,chrom_length):
    # pop=pop(getishuliang,u_num,jiyinchangdu)
    up=[]
    for i in range(pop_size):
        temp = []
        for j in range(u_num):
            t=0
            for k in range(chrom_length):
                t+=pop[i][j][k]*(math.pow(2,k))
            # print(t)
            u = t / (math.pow(2, chrom_length) - 1)
            temp.append(u)
            t = 0
        up.append(temp)
    return(up)
################################################
#3.根据守恒方程把u值放入，评价每个个体水平
# def PingJiaGeTi(s,up,pop_size,u_num):
#     pingjiazhi=[]
#     u=up
#     for i in range(pop_size):
#         f1=0
#         f2=0
#         k=s
#         for j in range(t_p):
#             n_p[1_1][k + 1] = n_p[1_1][k] + C * (d[1_1][k]) + t_c * (u[i][j] * m[2_1][k] - m[1_1][k])  # 调用decodechrom(pop,chrom_length)后u的前20个值是1—2，后20个值是2—1
#             n_p[1_2][k + 1] = n_p[1_2][k] + C * (d[1_2][k]) - t_c * (u[i][j+t_p] * m[1_2][k])
#             n_p[2_2][k + 1] = n_p[2_2][k] + C * (d[2_2][k]) + t_c * (u[i][j+t_p] * m[1_2][k] - m[2_2][k])
#             n_p[2_1][k + 1] = n_p[2_1][k] + C * (d[2_1][k]) - t_c * (u[i][j] * m[2_1][k])
#             if n_p[1_1][k + 1] < 0:
#                 n_p[1_1][k + 1] = 0
#             elif n_p[1_2][k + 1] < 0:
#                 n_p[1_2][k + 1] = 0
#             elif n_p[2_2][k + 1] < 0:
#                 n_p[2_2][k + 1] = 0
#             elif n_p[2_1][k + 1] < 0:
#                 n_p[2_1][k + 1] = 0
#
#             N_p['区域1'][k + 1] = n_p[1_1][k + 1] + n_p[1_2][k + 1]
#             N_p['区域2'][k + 1] = n_p[2_1][k + 1] + n_p[2_2][k + 1]
#
#             G['区域1'][k + 1] = -3E-09 * pow(N_p['区域1'][k + 1], 2) + 6E-05 * (N_p['区域1'][k + 1]) + 0.0183
#             G['区域2'][k + 1] = -6E-09 * pow(N_p['区域2'][k + 1], 2) + 8E-05 * (N_p['区域2'][k + 1]) + 0.0399
#
#             # print(vc)
#             if G['区域1'][k + 1] < 0:
#                 G['区域1'][k + 1] = 0
#             if G['区域2'][k + 1] < 0:
#                 G['区域2'][k + 1] = 0
#             if N_p['区域1'][k + 1] == 0 or G['区域1'][k + 1] == 0:
#                 m[1_1][k + 1] = 0
#                 m[1_2][k + 1] = 0
#             else:
#                 m[1_1][k + 1] = (n_p[1_1][k + 1] / N_p['区域1'][k + 1]) * G['区域1'][k + 1]
#                 m[1_2][k + 1] = (n_p[1_2][k + 1] / N_p['区域1'][k + 1]) * G['区域1'][k + 1]
#             if N_p['区域2'][k + 1] == 0 or G['区域2'][k + 1] == 0:
#                 m[2_1][k + 1] = 0
#                 m[2_2][k + 1] = 0
#             else:
#                 m[2_1][k + 1] = (n_p[2_1][k + 1] / N_p['区域2'][k + 1]) * G['区域2'][k + 1]
#                 m[2_2][k + 1] = (n_p[2_2][k + 1] / N_p['区域2'][k + 1]) * G['区域2'][k + 1]
#
#             if N_p['区域1'][k + 1] > 0:
#                 vc['区域1'][k + 1] = G['区域1'][k + 1] / (N_p['区域1'][k + 1] / (l[0] * 1000))
#             else:
#                 vc['区域1'][k + 1] = 16.7  # 限速60
#             if N_p['区域2'][k + 1] > 0:
#                  vc['区域2'][k + 1] = G['区域2'][k + 1] / (N_p['区域2'][k + 1] / (l[1] * 1000))
#             else:
#                 vc['区域2'][k + 1] = 16.7  # 限速60
#
#             if vc['区域1'][k + 1] < 0:
#                 vc['区域1'][k + 1] = 0
#             if vc['区域1'][k + 1] > 16.7:
#                 vc['区域1'][k + 1] = 16.7
#             if vc['区域2'][k + 1] < 0:
#                vc['区域2'][k + 1] = 0
#             if vc['区域2'][k + 1] > 16.7:
#                  vc['区域2'][k + 1] = 16.7
#
#             f1 += N_p['区域1'][k + 1] + N_p['区域2'][k + 1]
#             f2 += m[2_2][k + 1] + m[1_1][k + 1]  # 由于需求设置使得G为0导致m为0，故f2为0，存在问题（可能是G的函数存在问题）
#             k=k+1
#
#         f = f2 / f1
#         # print(f)
#         pingjiazhi.append(f)
#     return pingjiazhi


def PingJiaGeTi(pop, chrom_length,curent_t):  ### 通过MPC获得预测步数内的区域内累计车辆数和完成流，以m/N的最大值作为适应度
    pingjiazhi = []
    u = jiema(pop,pop_size,u_num,chrom_length)#40个个体，每个个体含有20个值，前10个为U21，后10个为U12
    # print('len(u)',len(u))
    for i in range(len(u)):
        # print(u[i])
        f1 = 0
        f2 = 0
        for k in range(curent_t, curent_t + t_p):
            n_p[1_1][k + 1] = n_p[1_1][k] + C * (d[1_1][k]) + t_c * (u[i][k - curent_t + 5] * m[2_1][k] - m[1_1][k])  # 调用decodechrom(pop,chrom_length)后u的前20个值是1—2，后20个值是2—1
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

            G['区域1'][k + 1] = -3E-09 * pow(N_p['区域1'][k + 1], 2) + 6E-05 * (N_p['区域1'][k + 1]) + 0.0183
            G['区域2'][k + 1] = -6E-09 * pow(N_p['区域2'][k + 1], 2) + 8E-05 * (N_p['区域2'][k + 1]) + 0.0399

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

            f1 += (N_p['区域1'][k + 1]-N1_C)**2
                  # + (N_p['区域2'][k + 1]-N2_C)**2
            # f2 += m[2_2][k + 1] + m[1_1][k + 1]  # 由于需求设置使得G为0导致m为0，故f2为0，存在问题（可能是G的函数存在问题）

        f = 1/f1
        pingjiazhi.append(f)
    # print(obj_value)
    # print(len(obj_value))

    return pingjiazhi


def calfiValue(pingjiazhi):
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


def sum(fit_value):
    total = 0
    for i in range(len(fit_value)):
        total += fit_value[i]
    return total


def cumsum(newfit):
    for i in range(len(newfit) - 2, -1, -1):
        t = 0
        j = 0
        while (j <= i):
            t += newfit[j]
            j += 1
        newfit[i] = t
        newfit[len(newfit) - 1] = 1

def selection(pop, fit_value):
    ms = []
    for i in range(len(pop)):
        ms.append(random.random())
    ms.sort()
    # print(ms)

    newfit=[]
    totalfit=sum(fit_value)
    for i in range(len(fit_value)):
        newfit.append(fit_value[i] / totalfit)
    # print(newfit)
    cumsum(newfit)

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


def crossover(pop, pc):
    for i in range(len(pop)-1):
        for j in range (u_num):
         if (random.random() < pc):
            cpoint = random.randint(0, jingdu(0,1))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][j][0:cpoint])
            temp1.extend(pop[i + 1][j][cpoint:len(pop[i][j])])
            temp2.extend(pop[i + 1][j][0:cpoint])
            temp2.extend(pop[i + 1][j][cpoint:len(pop[i][j])])
            pop[i][j] = temp1
            pop[i + 1][j] = temp2
    return pop


def mutation(pop, pm):
    for i in range(len(pop)):
        for j in range(u_num):
         if (random.random() < pm):
            mpoint = random.randint(0, jingdu(0,1) - 1)
            if (pop[i][j][mpoint] == 1):
                pop[i][j][mpoint] = 0
            else:
                pop[i][j][mpoint] = 1
    return pop
    # 找出最优值
def best(pop, fit_value):
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
def b2d(best_individual):
            length = len(best_individual)
            u_best = []
            t = 0
            for i in range(int(2 * t_p)):
                for k in range(jingdu(0,1)):
                    t += best_individual[i][k] * (math.pow(2, k))
                u = t / (math.pow(2, chrom_length) - 1)
                u_best.append(u)
                t = 0
            a = [u_best[0], u_best[5]]
            return a  # u_best[0]是在t时刻u12的最优值，u_best[19]是在t时刻u21的最优值

def result():
    pop=pop1
#这个数字代表种群繁衍代数
    for i in range(1):
        fit = PingJiaGeTi(pop, chrom_length)  # 从0时刻开始，计算30套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
        fitvalue = calfiValue(fit)  # 30个适应值
        # print(fitvalue)
        # best_individual, best_fit = best(pop, fitvalue)
        # u3 = b2d(best_individual)
        # print(u3)
        # print()
        pop = selection(pop, fitvalue)  # 30套方案，种群选择
        pop = crossover(pop, pc)
        pop = mutation(pop, pm)
    fit = PingJiaGeTi(pop, chrom_length)  # 从0时刻开始，计算30套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
    fitvalue = calfiValue(fit)  # 30个适应值
    # print(fitvalue)
    best_individual, best_fit = best(pop, fitvalue)
    u2 = b2d(best_individual)
    for i in range(2):
        if u2[i] <= 0.1:
            u2[i] = 0.1
    for i in range(2):
        if u2[i] >1:
            u2[i] = 1
    U21.append(u2[0])
    U12.append(u2[1])


    return U12,U21







if __name__ == "__main__":
    n_p[1_1][0] = 80
n_p[1_2][0] = 50
n_p[2_1][0] = 50
n_p[2_2][0] = 40
N_p['区域1'][0] = 130
N_p['区域2'][0] = 90
pc = 0.8
pm = 0.05
chrom_length=jingdu(0,1)
print(chrom_length)
pop_size=40  #40个个体
u_num=10  #每个个体有20个值
t_s=300 #控制步数为300步
U21=[1]
U12=[1]
s=1
x=[1]

M1_2=[0]
M2_1=[0]

for j in range(20):
    if_stop = False  # 是否停止循环，差值小于0.5时停止
    while not if_stop:
        curent_t=j
        pop = pop1(pop_size, u_num, chrom_length)  # 产生初始种群
        for i in range(30):
            fit = PingJiaGeTi(pop, chrom_length,curent_t)  # 从0时刻开始，计算40套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
            fitvalue = calfiValue(fit)  # 40个适应值
            # print('len(fitvalue)',len(fitvalue))
            # best_individual, best_fit = best(pop, fitvalue)
            # u3 = b2d(best_individual)
            # print('len(u3)',len(u3))
            # # print()
            pop = selection(pop, fitvalue)  # 30套方案，种群选择
            pop = crossover(pop, pc)
            pop = mutation(pop, pm)
        fit = PingJiaGeTi(pop, chrom_length,curent_t)  # 从0时刻开始，计算30套方案的适应函数#3.根据守恒方程把u值放入，评价每个个体水平PingJiaGeTi(s,up,pop_size,u_num)
        fitvalue = calfiValue(fit)  # 40个适应值
        # print(fitvalue)
        best_individual, best_fit = best(pop, fitvalue)
        u2 = b2d(best_individual)
        for i in range(2):
            if u2[i] <= 0.1:
                u2[i] = 0.1
        for i in range(2):
            if u2[i] >1:
                u2[i] = 1
        off_u21 = abs(u2[0]-U21[-1])
        if off_u21<=0.3:
          U21.append(u2[0])
          U12.append(u2[1])
          com21 = (U21[0] * m[2_1][curent_t]) * t_c
          com12 = (U12[1] *m[1_2][curent_t]) * t_c
          M1_2.append(com21)
          M2_1.append(com12)
        # result()
          s=s+1
          x.append(s)
          if_stop = True
        else:
            pass


print(U21)
print(U12)
print(M1_2)
print(M2_1)
# #
# #
plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('time(5min)')
plt.ylabel('U')
plt.plot(x,U21,label='u21',linewidth=1)
plt.plot(x,U12,label='u12',linewidth=0.5)
plt.legend(loc='upper left')
plt.title('边界控制')
plt.grid(True)
plt.show()

plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('time(5min)')
plt.ylabel('U')
plt.plot(x,M2_1,label='M2_1',linewidth=1)
plt.plot(x,M1_2,label='M1_2',linewidth=0.5)
plt.legend(loc='upper left')
plt.title('实际转移值')
plt.grid(True)
plt.show()


