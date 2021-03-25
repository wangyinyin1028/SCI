import random
import math
import numpy as np

#############################  second_layer 参数 #############################################
l1 =[3.1,2.9,3.0,3.3]#四个路口的长度
l2 =[1.9,2.1,2.0,1.3]
lh = [1.2,1.35,1.5]        # 交叉口之间的距离，单位 km
alpha = [0.3,0.5,0.2]      # 0.3左转，0.5直行，0.2右转
ps = 80
lv = 4                         # 私家车长度 ， 单位m
n_m = {1: np.zeros((900,4)), 2: np.zeros((900,4)), 3: np.zeros((900,4)), 4: np.zeros((900,4))}#交叉口进口等待的社会车与公交车之和
n_mb = {1: np.zeros((900,4)), 2: np.zeros((900,4)), 3: np.zeros((900,4)), 4: np.zeros((900,4))}
n_mc = {1: np.zeros((900,4)), 2: np.zeros((900,4)), 3: np.zeros((900,4)), 4: np.zeros((900,4))}

td_c = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#私家车时延
td_b = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#公交车时延

i_m = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#交叉口进口的社会车与公交车之和
i_mb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#四个进口道公交车的等待车辆数
i_mc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#四个进口道私家车的等待车辆数

vb = 11             # 公交速度 单位m/s
h_list = []
# r1 = np.zeros((900,4))           # r1是区域1社会车诱导参数，通过logit模型计算，这里为了试验直接给定值
# r2 = np.zeros((900,4))           # r2是区域2社会车诱导参数，通过logit模型计算，这里为了试验直接给定值
# r = 0.25
r1 = np.zeros((900,4))
r2 = np.zeros((900,4))
sta_flow = 0.8         # 饱和流
qb1 = 0.01
qc1 = 0.05
qb2 = 0.01
qc2 = 0.05

l_mb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#左转
l_mc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
l_m = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#四个进口道的左转车辆数
s_mb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
s_mc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#直行
s_m = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}

o_m = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#交叉口出口的社会车与公交车之和
o_mb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#四个进口道公交车的出去车辆数
o_mc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#四个进口道私家车的出去车辆数
o = []
obj = []
         # 非控制区域到达交叉口的到达率

d_mlb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}#人均等待时间
d_mlc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
d_msb = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
d_msc = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
d_hb = np.zeros((900,4))
d_hc = np.zeros((900,4))
d_hm = {'出口1': np.zeros((900,4)), '出口2': np.zeros((900,4)), '出口3': np.zeros((900,4)), '出口4': np.zeros((900,4))}
# d_h = np.zeros((900,4))
num_b = 20
num_c = 1.5

beta1 = [0.25,0.25,0.25,0.25]
beta2 = [0.25,0.25,0.25,0.25]




nnnnn = []
######################################################### second_layer 参数 ###########################################
def init():
    pq = 160
    f = 0.5
    cr = 0.8
    gen = 1000
    len_g = 8
    g_min = 20
    g_max = 60
    return pq, f, cr, gen, len_g, g_min, g_max


def initialtion2(pq):
    len = int(len_g / 2)
    pq1_list = []
    pq2_list = []
    for i in range(pq):
        g_l_list = np.zeros(8)
        g_s_list = np.zeros(8)
        for j in range(len):
            g_l_list[j] = (int(20 + random.random() * (60 - 20)))
            g_l_list[j + 4] = (int(20 + random.random() * (60 - 20)))
            # g_l_list[j+4] = (int(random.randint(20,60)))
            g_s_list[j] = (int(20 + random.random() * (60 - 20)))
            g_s_list[j + 4] = (int(120 - g_l_list[j] - g_l_list[j + 4] - g_s_list[j]))
        for x in range(4, 8):
            while g_s_list[x] < 20:
                g_l_list[x - 4] = (int(20 + random.random() * (60 - 20)))
                g_l_list[x] = (int(20 + random.random() * (60 - 20)))
                g_s_list[x - 4] = (int(20 + random.random() * (60 - 20)))
                g_s_list[x] = (int(120 - g_l_list[x - 4] - g_l_list[x] - g_s_list[x - 4]))

        pq1_list.append(g_l_list)
        pq2_list.append(g_s_list)

    return pq1_list, pq2_list
cha=[]
C=120
def objective2(g_l, g_s):
    emi1 = 0
    emi2 = 0
    nm11 = 0
    nm22 = 0
    nm33 = 0
    nm44 = 0
    n1111 = []
    n2222 = []
    n3333 = []
    n4444 = []

    r_total1 = 0
    r_total2 = 0

    d_h = []
    t=0#实验当前时刻为0时刻，一开始路段上车辆数为0，
    #交叉口1各进口等待车辆数,j是交叉口进口道,h是交叉口
    # for j in range(4):
    #     for h in range(4):
    #         n_m[j][t][h]#t时刻，h交叉口，j进口道的等待车辆数
    for h in range(4):
        td_b['出口1'][t][h]=math.ceil(((l1[0]-ps)/vb)/C)
        td_c['出口1'][t][h] = math.ceil(((l1[0]*1000*3 - n_m[1][t][h]*lv) / 3*vc['区域1'][0] * C) )#为什么这里的社会车速度不是等于实时的区域平均速度
        td_b['出口3'][t][h] = math.ceil(((l1[0] - ps) / vb) / C)
        td_c['出口3'][t][h] = math.ceil((l1[0] * 1000 * 3 - n_m[1][t][h] * lv) / (3 * vc['区域1'][0] * C))# 为什么这里的社会车速度不是等于实时的区域平均速度
    if h == 0:
        td_b['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
        td_c['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
    else:
        td_b['出口2'][t][h] = math.ceil(((lh[h - 1] * 1000 - ps) / vb) / C)
        td_c['出口2'][t][h] = math.ceil(((lh[h - 1] * 1000 * 3 - n_m['出口2'][t][h] * lv - ps) / (3 * vc['区域1'][0] * C)))  # 实际上出口2、出口4的速度跟区域的速度不一样
    if h == 3:
        td_b['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
        td_c['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
    else:
        td_b['出口4'][t][h] = math.ceil(((lh[h] * 1000 - ps) / vb) / C)
        td_c['出口4'][t][h] = math.ceil(((lh[h] * 1000 * 3 - n_m['出口4'][t][h] * lv - ps) / (3 * vc['区域2'][0] * C)))
        # 计算进入交叉口的车辆数，每个交叉口4个进口，4*2种时延情况
        cb1 = int(t - abs(td_b['出口1'][t][h]))
        cc1 = int(t - abs(td_c['出口1'][t][h]))

        cb2 = int(t - abs(td_b['出口3'][t][h]))
        cc2 = int(t - abs(td_c['出口3'][t][h]))

        cb3 = int(t - abs(td_b['出口2'][t][h]))
        cc3 = int(t - abs(td_c['出口2'][t][h]))

        cb4 = int(t - abs(td_b['出口4'][t][h]))
        cc4 = int(t - abs(td_c['出口4'][t][h]))
        ### 四个交叉口中，进口1和进口3的上游不存在交叉口，所以进入该交叉口的车辆数通过需求获得，进口2和进口4可能存在交叉口也可能不存在交叉口，所以进行分类计算
        if cb1 >= 0 and cb2 >= 0 and cb3 >= 0 and cb4 >= 0 and cc1 >= 0 and cc2 >= 0 and cc3 >= 0 and cc4 >= 0:
            i_mb['出口1'][t][h] = C * (db[1_2][cb1] / 4)  # 公交车均分至四个路口
            i_mc['出口1'][t][h] = C * (r1[cc1][h] * dc[1_2][cc1])  # 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
            i_m['出口1'][t][h] = i_mb['出口1'][t][h] + i_mc['出口1'][t][h]  # 交叉口进口的社会车与公交车之和
            i_mb['出口3'][t][h] = C * (db[2_1][cb2] / 4)
            i_mc['出口3'][t][h] = C * (r2[cc2][h] * dc[2_1][cc2])
            i_m['出口3'][t][h] = i_mb['出口3'][t][h] + i_mc['出口3'][t][h]
            if h == 0:
                i_mb['出口2'][t][h] = C * qb1  # 非控制区域3的需求，直接通过到达率计算
                i_mc['出口2'][t][h] = C * qc1  # 非控制区域3的需求，直接通过到达率计算
                i_m['出口2'][t][h] = i_mb['出口2'][t][h] + i_mc['出口2'][t][h]
            else:
                i_mb['出口2'][t][h] = o_mb['出口1'][cb3][h - 1] * alpha[2] + o_mb['出口2'][cb3][h - 1] * alpha[1] + \
                                    o_mb['出口3'][cb3][h - 1] * alpha[0]  # 存疑 ，为什么还是alpha0,
                i_mc['出口2'][t][h] = o_mc['出口1'][cc3][h - 1] * alpha[2] + o_mc['出口2'][cc3][h - 1] * alpha[1] + \
                                    o_mc['出口3'][cc3][h - 1] * alpha[0]  # 存疑 ，
                i_m['出口2'][t][h] = i_mb['出口2'][t][h] + i_mc['出口2'][t][h]
            if h == 3:
                i_mb['出口4'][t][h] = C * qb2  # 非控制区域4的需求，直接通过到达率计算
                i_mc['出口4'][t][h] = C * qc2  # 非控制区域4的需求，直接通过到达率计算
                i_m['出口4'][t][h] = i_mb['出口4'][t][h] + i_mc['出口4'][t][h]
            else:
                i_mb['出口4'][t][h] = o_mb['出口1'][cb4][h + 1] * alpha[0] + o_mb['出口3'][cb4][h + 1] * alpha[2] + \
                                    o_mb['出口4'][cb4][h + 1] * alpha[1]
                i_mc['出口4'][t][h] = o_mc['出口1'][cc4][h + 1] * alpha[0] + o_mc['出口3'][cc4][h + 1] * alpha[2] + \
                                    o_mc['出口4'][cc4][h + 1] * alpha[1]
                i_m['出口4'][t][h] = i_mb['出口4'][t][h] + i_mc['出口4'][t][h]
        if np.all((n_m['出口1'][t][h] * alpha[0] + n_m['出口3'][t][h] * alpha[0]) < g_l[h] * sta_flow * 2):
            l_mb['出口1'][t][h] = n_mb['出口1'][t][h] * alpha[0]
            l_mc['出口1'][t][h] = n_mc['出口1'][t][h] * alpha[0]  # z这里应该是n_mc
            l_m['出口1'][t][h] = l_mb['出口1'][t][h] + l_mc['出口1'][t][h]
            l_mb['出口3'][t][h] = n_m['出口3'][t][h] * alpha[0]
            l_mc['出口3'][t][h] = n_m['出口3'][t][h] * alpha[0]  # 这里不区分公交车与社会车么？？
            l_m['出口3'][t][h] = l_mb['出口3'][t][h] + l_mc['出口3'][t][h]
            d_mlb['出口1'][t + 1][h] = 0
            d_mlc['出口1'][t + 1][h] = 0
            d_mlb['出口3'][t + 1][h] = 0
            d_mlc['出口3'][t + 1][h] = 0
        else:
            l_mb['出口1'][t][h] = round((n_mb['出口1'][t][h] / n_m['出口1'][t][h]) * g_l[h] * sta_flow)  # 等待车辆数中的公交车与社会车占比进行排放
            l_mc['出口1'][t][h] = round((n_mc['出口1'][t][h] / n_m['出口1'][t][h]) * g_l[h] * sta_flow)
            l_m['出口1'][t][h] = l_mb['出口1'][t][h] + l_mc['出口1'][t][h]
            l_mb['出口3'][t][h] = round((n_mb['出口3'][t][h] / n_m['出口3'][t][h]) * g_l[h] * sta_flow)
            l_mc['出口3'][t][h] = round((n_mc['出口3'][t][h] / n_m['出口3'][t][h]) * g_l[h] * sta_flow)
            l_m['出口3'][t][h] = l_mb['出口3'][t][h] + l_mc['出口3'][t][h]
            d_mlb['出口1'][t + 1][h] = math.ceil((alpha[0] * n_mb['出口1'][t][h] - l_mb['出口1'][t][h]) / g_l[h] * sta_flow) * C * (alpha[0] * n_mb['出口1'][t][h] - l_mb['出口1'][t][h]) * num_b
            d_mlc['出口1'][t + 1][h] = math.ceil((alpha[0] * n_mc['出口1'][t][h] - l_mc['出口1'][t][h]) / g_l[h] * sta_flow) * C * (alpha[0] * n_mc['出口1'][t][h] - l_mc['出口1'][t][h]) * num_c
            d_mlb['出口3'][t + 1][h] = math.ceil((alpha[0] * n_mb['出口3'][t][h] - l_mb['出口3'][t][h]) / g_l[h] * sta_flow) * C * (alpha[0] * n_mb['出口3'][t][h] - l_mb['出口3'][t][h]) * num_b
            d_mlc['出口3'][t + 1][h] = math.ceil((alpha[0] * n_mc['出口3'][t][h] - l_mb['出口3'][t][h]) / g_l[h] * sta_flow) * C * (alpha[0] * n_mc['出口3'][t][h] - l_mc['出口3'][t][h]) * num_c
        if (n_m['出口1'][t][h] * alpha[0] + n_m['出口3'][t][h]) * (1 - alpha[0]) < g_s[h] * sta_flow * 2:
            s_mb['出口1'][t][h] = n_mb['出口1'][t][h] * (1 - alpha[0])  # 为什么没考虑右转呢？
            s_mc['出口1'][t][h] = n_mc['出口1'][t][h] * (1 - alpha[0])
            s_m['出口1'][t][h] = s_mb['出口1'][t][h] + s_mc['出口1'][t][h]
            s_mb['出口3'][t][h] = n_mb['出口3'][t][h] * (1 - alpha[0])
            s_mc['出口3'][t][h] = n_mc['出口3'][t][h] * (1 - alpha[0])
            s_m['出口3'][t][h] = s_mb['出口3'][t][h] + s_mc['出口3'][t][h]
            d_msb['出口1'][t + 1][h] = s_mb['出口1'][t][h] * num_b * g_l[h]#等待时间为左转绿灯时间
            d_msc['出口1'][t + 1][h] = s_mc['出口1'][t][h] * num_c * g_l[h]
            d_msb['出口3'][t + 1][h] = s_mb['出口3'][t][h] * num_b * g_l[h]
            d_msc['出口3'][t + 1][h] = s_mc['出口3'][t][h] * num_c * g_l[h]
        else:
            # s['1&3'][t][h] = g_s['1&3'][t][h]*sta_flow*2
            s_mb['出口1'][t][h] = round(n_mb['出口1'][t][h] / n_m['出口1'][t][h]) * g_s[h] * sta_flow
            s_mc['出口1'][t][h] = round(n_mc['出口1'][t][h] / n_m['出口1'][t][h]) * g_s[h] * sta_flow
            s_m['出口1'][t][h] = s_mb['出口1'][t][h] + s_mc['出口1'][t][h]
            s_mb['出口3'][t][h] = round(n_mb['出口3'][t][h] / n_m['出口3'][t][h]) * g_s[h] * sta_flow
            s_mc['出口3'][t][h] = round(n_mc['出口3'][t][h] / n_m['出口3'][t][h]) * g_s[h] * sta_flow
            s_m['出口3'][t][h] = s_mb['出口3'][t][h] + s_mc['出口3'][t][h]
            d_msb['出口1'][t + 1][h] = s_mb['出口1'][t][h] * num_b * g_l[h] + math.ceil((n_mb['出口1'][t][h] * (1 - alpha[0]) - s_mb['出口1'][t][h]) / g_s[h] * sta_flow) * C * (n_mb['出口1'][t][h] * (1 - alpha[0]) - s_mb['出口1'][t][h]) * num_b
            d_msc['出口1'][t + 1][h] = s_mc['出口1'][t][h] * num_c * g_l[h] + math.ceil((n_mc['出口1'][t][h] * (1 - alpha[0]) - s_mc['出口1'][t][h]) / g_s[h] * sta_flow) * C * (n_mc['出口1'][t][h] * (1 - alpha[0]) - s_mc['出口1'][t][h]) * num_c
            d_msb['出口3'][t + 1][h] = s_mb['出口3'][t][h] * num_b * g_l[h] + math.ceil((n_mb['出口3'][t][h] * (1 - alpha[0]) - s_mb['出口3'][t][h]) / g_s[h] * sta_flow) * C * (n_mb['出口3'][t][h] * (1 - alpha[0]) - s_mb['出口3'][t][h]) * num_b
            d_msc['出口3'][t + 1][h] = s_mc['出口3'][t][h] * num_c * g_l[h] + math.ceil((n_mc['出口3'][t][h] * (1 - alpha[0]) - s_mc['出口3'][t][h]) / g_s[h] * sta_flow) * C * (n_mc['出口3'][t][h] * (1 - alpha[0]) - s_mc['出口3'][t][h]) * num_c
        if d_mlb['出口1'][t][h] > 0 and d_mlc['出口1'][t][h] > 0 and d_msb['出口1'][t][h] > 0 and d_msc['出口1'][t][h] > 0:
            d_hm['出口1'][t][h] = ((d_mlb['出口1'][t][h] + d_mlc['出口1'][t][h] + d_msb['出口1'][t][h] + d_msc['出口1'][t][h]) / (
                        n_mb['出口1'][t][h] * num_b + n_mc['出口1'][t][h] * num_c))
        else:
            d_hm['出口1'][t][h] = 0
        if d_mlb['出口3'][t][h] > 0 and d_mlc['出口3'][t][h] > 0 and d_msb['出口3'][t][h] > 0 and d_msc['出口3'][t][h] > 0:
            d_hm['出口3'][t][h] = ((d_mlb['出口3'][t][h] + d_mlc['出口3'][t][h] + d_msb['出口3'][t][h] + d_msc['出口3'][t][h]) / (
                        n_mb['出口3'][t][h] * num_b + n_mc['出口3'][t][h] * num_c))
        else:
            d_hm['出口3'][t][h] = (0)
        o_mb['出口1'][t][h] = l_mb['出口1'][t][h] + s_mb['出口1'][t][h]
        o_mc['出口1'][t][h] = l_mc['出口1'][t][h] + s_mc['出口1'][t][h]
        o_m['出口1'][t][h] = o_mb['出口1'][t][h] + o_mc['出口1'][t][h]
        o_mb['出口3'][t][h] = l_mb['出口3'][t][h] + s_mb['出口3'][t][h]
        o_mc['出口3'][t][h] = l_mc['出口3'][t][h] + s_mc['出口3'][t][h]
        o_m['出口3'][t][h] = o_mb['出口3'][t][h] + o_mc['出口3'][t][h]
        if (n_m['出口2'][t][h] * alpha[0] + n_m['出口4'][t][h] * alpha[0]) < g_l[h + 4] * sta_flow * 2:
            l_mb['出口2'][t][h] = n_mb['出口2'][t][h] * alpha[0]
            l_mc['出口2'][t][h] = n_mc['出口2'][t][h] * alpha[0]
            l_m['出口2'][t][h] = l_mb['出口2'][t][h] + l_mc['出口2'][t][h]
            l_mb['出口4'][t][h] = n_mb['出口4'][t][h] * alpha[0]
            l_mc['出口4'][t][h] = n_mc['出口4'][t][h] * alpha[0]
            l_m['出口4'][t][h] = l_mb['出口4'][t][h] + l_mc['出口4'][t][h]
            d_mlb['出口2'][t + 1][h] = l_mb['出口2'][t][h] * (g_l[h] + g_s[h]) * num_b
            d_mlc['出口2'][t + 1][h] = l_mc['出口2'][t][h] * (g_l[h] + g_s[h]) * num_c
            d_mlb['出口4'][t + 1][h] = l_mb['出口4'][t][h] * (g_l[h] + g_s[h]) * num_b
            d_mlc['出口4'][t + 1][h] = l_mc['出口4'][t][h] * (g_l[h] + g_s[h]) * num_c
        else:
            # l['2&4'][t][h] = g_l['2&4'][t][h] * sta_flow * 2
            l_mb['出口2'][t][h] = round(n_mb['出口2'][t][h] / n_m['出口2'][t][h]) * g_l[h + 4] * sta_flow
            l_mc['出口2'][t][h] = round(n_mc['出口2'][t][h] / n_m['出口2'][t][h]) * g_l[h + 4] * sta_flow
            l_m['出口2'][t][h] = l_mb['出口2'][t][h] + l_mc['出口2'][t][h]
            l_mb['出口4'][t][h] = round(n_mb['出口4'][t][h] / n_m['出口4'][t][h]) * g_l[h + 4] * sta_flow
            l_mc['出口4'][t][h] = round(n_mc['出口4'][t][h] / n_m['出口4'][t][h]) * g_l[h + 4] * sta_flow
            l_m['出口4'][t][h] = l_mb['出口4'][t][h] + l_mc['出口4'][t][h]
            d_mlb['出口2'][t + 1][h] = l_mb['出口2'][t][h] * (g_l[h] + g_s[h]) * num_b + math.ceil(
                (n_mb['出口2'][t][h] * alpha[0] - l_mb['出口2'][t][h]) / (g_l[h + 4] * sta_flow)) * C * (
                                                 n_mb['出口2'][t][h] * alpha[0] - l_mb['出口2'][t][h]) * num_b
            d_mlc['出口2'][t + 1][h] = l_mc['出口2'][t][h] * (g_l[h] + g_s[h]) * num_c + math.ceil(
                (n_mc['出口2'][t][h] * alpha[0] - l_mc['出口2'][t][h]) / (g_l[h + 4] * sta_flow)) * C * (
                                                 n_mc['出口2'][t][h] * alpha[0] - l_mc['出口2'][t][h]) * num_c
            d_mlb['出口4'][t + 1][h] = l_mb['出口4'][t][h] * (g_l[h] + g_s[h]) * num_b + math.ceil(
                (n_mb['出口4'][t][h] * alpha[0] - l_mb['出口4'][t][h]) / (g_l[h + 4] * sta_flow)) * C * (
                                                 n_mb['出口4'][t][h] * alpha[0] - l_mb['出口4'][t][h]) * num_b
            d_mlc['出口4'][t + 1][h] = l_mc['出口4'][t][h] * (g_l[h] + g_s[h]) * num_c + math.ceil(
                (n_mc['出口4'][t][h] * alpha[0] - l_mc['出口4'][t][h]) / (g_l[h + 4] * sta_flow)) * C * (
                                                 n_mc['出口4'][t][h] * alpha[0] - l_mc['出口4'][t][h]) * num_c
        if (n_m['出口2'][t][h] * alpha[0] + n_m['出口4'][t][h]) * (1 - alpha[0]) < g_s[h + 4] * sta_flow * 2:
            # s['2&4'][t][h] = (n_m['出口2'][t][h] * alpha[0] + n_m['出口4'][t][h]) * (1 - alpha[0])
            s_mb['出口2'][t][h] = n_mb['出口2'][t][h] * (1 - alpha[0])
            s_mc['出口2'][t][h] = n_mc['出口2'][t][h] * (1 - alpha[0])
            s_m['出口2'][t][h] = s_mb['出口2'][t][h] + s_mc['出口2'][t][h]
            s_mb['出口4'][t][h] = n_mb['出口4'][t][h] * (1 - alpha[0])
            s_mc['出口4'][t][h] = n_mc['出口4'][t][h] * (1 - alpha[0])
            s_m['出口4'][t][h] = s_mb['出口4'][t][h] + s_mc['出口4'][t][h]
            d_msb['出口2'][t + 1][h] = s_mb['出口2'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_b
            d_msc['出口2'][t + 1][h] = s_mc['出口2'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_c
            d_msb['出口4'][t + 1][h] = s_mb['出口4'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_b
            d_msc['出口4'][t + 1][h] = s_mc['出口4'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_c
        else:
            # s['2&4'][t][h] = g_s['2&4'][t][h] * sta_flow * 2
            s_mb['出口2'][t][h] = round(n_mb['出口2'][t][h] / n_m['出口2'][t][h]) * g_s[
                h + 4] * sta_flow  # 在没有预信号的时候，公交与社会车排放按等待车辆中的比例
            s_mc['出口2'][t][h] = round(n_mc['出口2'][t][h] / n_m['出口2'][t][h]) * g_s[h + 4] * sta_flow
            s_m['出口2'][t][h] = s_mb['出口2'][t][h] + s_mc['出口2'][t][h]
            s_mb['出口4'][t][h] = round(n_mb['出口4'][t][h] / n_m['出口4'][t][h]) * g_s[h + 4] * sta_flow
            s_mc['出口4'][t][h] = round(n_mc['出口4'][t][h] / n_m['出口4'][t][h]) * g_s[h + 4] * sta_flow
            s_m['出口4'][t][h] = s_mb['出口4'][t][h] + s_mc['出口4'][t][h]
            d_msb['出口2'][t + 1][h] = s_mb['出口2'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_b + math.ceil(
                (n_mb['出口2'][t][h] * (1 - alpha[0]) - s_mb['出口2'][t][h]) / (g_s[h + 4] * sta_flow)) * C * (
                                                 n_mb['出口2'][t][h] * (1 - alpha[0]) - s_mb['出口2'][t][h]) * num_b
            d_msc['出口2'][t + 1][h] = s_mc['出口2'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_c + math.ceil(
                (n_mc['出口2'][t][h] * (1 - alpha[0]) - s_mc['出口2'][t][h]) / (g_s[h + 4] * sta_flow)) * C * (
                                                 n_mc['出口2'][t][h] * (1 - alpha[0]) - s_mc['出口2'][t][h]) * num_c
            d_msb['出口4'][t + 1][h] = s_mb['出口4'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_b + math.ceil(
                (n_mb['出口4'][t][h] * (1 - alpha[0]) - s_mb['出口4'][t][h]) / (g_s[h + 4] * sta_flow)) * C * (
                                                 n_mb['出口4'][t][h] * (1 - alpha[0]) - s_mb['出口4'][t][h]) * num_b
            d_msc['出口4'][t + 1][h] = s_mc['出口4'][t][h] * (g_l[h] + g_s[h] + g_l[h + 4]) * num_c + math.ceil(
                (n_mc['出口4'][t][h] * (1 - alpha[0]) - s_mc['出口4'][t][h]) / (g_s[h + 4] * sta_flow)) * C * (
                                                 n_mc['出口4'][t][h] * (1 - alpha[0]) - s_mc['出口4'][t][h]) * num_c
        if d_mlb['出口2'][t][h] > 0 and d_mlc['出口2'][t][h] > 0 and +d_msb['出口2'][t][h] > 0 and d_msc['出口2'][t][h] > 0:
            d_hm['出口2'][t][h] = ((d_mlb['出口2'][t][h] + d_mlc['出口2'][t][h] + d_msb['出口2'][t][h] + d_msc['出口2'][t][h]) / (
                        n_mb['出口2'][t][h] * num_b + n_mc['出口2'][t][h] * num_c))
        else:
            d_hm['出口2'][t][h] = 0
        if d_mlb['出口4'][t][h] > 0 and d_mlc['出口4'][t][h] > 0 and d_msb['出口4'][t][h] > 0 and d_msc['出口4'][t][h] > 0:
            d_hm['出口4'][t][h] = ((d_mlb['出口4'][t][h] + d_mlc['出口4'][t][h] + d_msb['出口4'][t][h] + d_msc['出口4'][t][h]) / (
                        n_mb['出口4'][t][h] * num_b + n_mc['出口4'][t][h] * num_c))
        else:
            d_hm['出口4'][t][h] = 0
        o_mb['出口2'][t][h] = l_mb['出口2'][t][h] + s_mb['出口2'][t][h]
        o_mc['出口2'][t][h] = l_mc['出口2'][t][h] + s_mc['出口2'][t][h]
        o_m['出口2'][t][h] = o_mb['出口2'][t][h] + o_mc['出口2'][t][h]
        o_mb['出口4'][t][h] = l_mb['出口4'][t][h] + s_mb['出口4'][t][h]
        o_mc['出口4'][t][h] = l_mc['出口4'][t][h] + s_mc['出口4'][t][h]
        o_m['出口4'][t][h] = o_mb['出口4'][t][h] + o_mc['出口4'][t][h]
        emi1 += l_m['出口2'][t][h] + s_m['出口4'][t][h] * 2 / 7 + s_m['出口1'][t][h] * 5 / 7  # 交叉口直行、左转、右转按5:3:2的比例，所以在s_m中有5/7是直行的车辆
        emi2 += s_m['出口2'][t][h] * 2 / 7 + l_m['出口4'][t][h] + s_m['出口3'][t][h] * 5 / 7  # 交叉口直行、左转、右转按5:3:2的比例，所以在s_m中有5/7是直行的车辆
    for h in range(4):
        d_h.append(d_hm['出口1'][t][h]+d_hm['出口2'][t][h]+d_hm['出口3'][t][h]+d_hm['出口4'][t][h])
    # 通过logit模型计算分流率
    for h in range(4):
            if n_m['出口1'][t][0] > 0 and n_m['出口1'][t][1] > 0 and n_m['出口1'][t][2] > 0 and n_m['出口1'][t][3] > 0 and n_m['出口3'][t][0] > 0 and n_m['出口3'][t][1] > 0 and n_m['出口3'][t][2] > 0 and n_m['出口3'][t][3] > 0:
                    r_total1 += math.log(0.2 * n_m['出口1'][t][h] / (n_m['出口1'][t][0] + n_m['出口1'][t][1] + n_m['出口1'][t][2] + n_m['出口1'][t][3]), math.e)
                    r_total2 += math.log(0.2 * n_m['出口3'][t][h] / (n_m['出口3'][t][0] + n_m['出口3'][t][1] + n_m['出口3'][t][2] + n_m['出口3'][t][3]), math.e)
    for h in range(4):
                if n_m['出口1'][t][0] > 0 and n_m['出口1'][t][1] > 0 and n_m['出口1'][t][2] > 0 and n_m['出口1'][t][3] > 0 and n_m['出口3'][t][0] > 0 and n_m['出口3'][t][1] > 0 and n_m['出口3'][t][2] > 0 and n_m['出口3'][t][3] > 0:
                    r1[t][h] = math.log(0.2*n_m['出口1'][t][h]/(n_m['出口1'][t][0] + n_m['出口1'][t][1] + n_m['出口1'][t][2]+n_m['出口1'][t][3]),math.e) / r_total1
                    r2[t][h] = math.log(0.2*n_m['出口3'][t][h]/(n_m['出口3'][t][0] + n_m['出口3'][t][1] + n_m['出口3'][t][2] + n_m['出口3'][t][3]),math.e) / r_total2
                else:
                    r1[t][h] = 0.25
                    r2[t][h] = 0.25



########################################## main ############################################
if __name__ == "__main__":
    pq = 10
f = 0.5
cr = 0.8
gen = 1000
len_g = 8
g_min = 20
g_max = 60
pq1_list, pq2_list = initialtion2(pq)#形成初始种群，种群大小为pq=10
# print(pq1_list)
# print(pq2_list)
objective2(pq1_list[0], pq2_list[0])