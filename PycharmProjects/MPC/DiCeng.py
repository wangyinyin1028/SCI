import numpy as np
import math
import random
import torch
import torch.nn as nn
from collections import deque
import torch.nn.functional as F                 # 导入torch.nn.functional
import random

#超参数
BATCH_SIZE = 32                                 # 样本数量
LR = 0.01                                       # 学习率
EPSILON = 0.9                                   # greedy policy
GAMMA = 0.9                                     # reward discount
TARGET_REPLACE_ITER = 100                       # 目标网络更新频率
MEMORY_CAPACITY = 2000                          # 记忆库容量
# 定义环境
N_ACTIONS = 9                                   # 动作个数 (6个)
N_STATES = 18

class env(object):
    def __init__(self):
        self.l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
        self.l2 = [1.9, 2.1, 2.0, 1.3]
        self.lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
        self.alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
        self.lv = 4  # 私家车长度 ， 单位m
        self.n_m={1: np.zeros((900, 4)), 2: np.zeros((900, 4)), 3: np.zeros((900, 4)),
           4: np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和
        self.td_c  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 私家车时延
        self.i_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车与公交车之和
        self.sta_flow  = 0.8  # 饱和流
        self.l_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数
        self.s_m  = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
        self.o_m  =  {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和
        self.qc1 = 0.05
        self.qc2 = 0.05
    C=120
    qc1 = 0.05
    qc2 = 0.05
    def Doaction(self,action,t):
       for h in range(4):
          self.td_c['出口1'][t][h] = math.ceil(((self.l1[0] * 1000 * 3 - self.n_m[1][t][h] * self.lv) / 3 * self.vc['区域1'][0] * C))  # 为什么这里的社会车速度不是等于实时的区域平均速度
          self.td_c['出口3'][t][h] = math.ceil(((self.l1[3] * 1000 * 3 - self.n_m[1][t][h] * self.lv) / 3 * self.vc['区域2'][0] * C))  # 为什么这里的社会车速度不是等于实时的区域平均速度
          if h == 0:
              self.td_c['出口2'][t][h] = 0  # 非控制区域3的需求，直接通过到达率计算
          else:
              self.td_c['出口2'][t][h] =math.ceil(((self.lh[h - 1] * 1000 * 3 - self.n_m['出口2'][t][h] * self.lv) / 3 * self.vc['区域1'][0] * C))
          if h == 3:
              self.td_b['出口4'][t][h] = 0  # 非控制区域4的需求，直接通过到达率计算
          else:
              self.td_c['出口4'][t][h] =math.ceil(((self.lh[h] * 1000 * 3 - self.n_m['出口4'][t][h] * self.lv) / 3 * self.vc['区域2'][0] * C))
              # 计算进入交叉口的车辆数，每个交叉口4个进口，4*2种时延情况
       cc1 = int(t - abs(self.td_c['出口1'][t][h]))
       cc2 = int(t - abs(self.td_c['出口3'][t][h]))
       cc3 = int(t - abs(self.td_c['出口2'][t][h]))
       cc4 = int(t - abs(self.td_c['出口4'][t][h]))
       ### 四个交叉口中，进口1和进口3的上游不存在交叉口，所以进入该交叉口的车辆数通过需求获得，进口2和进口4可能存在交叉口也可能不存在交叉口，所以进行分类计算
       if cc1 >= 0 and cc2 >= 0 and cc3 >= 0 and cc4 >= 0:
           self.i_mc['出口1'][t][h] = C * (r1[cc1][h] * d[1_2][cc1])  # 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
           self.i_mc['出口3'][t][h] = C * (r2[cc2][h] * d[2_1][cc2])
           if h == 0:
               self.i_mc['出口2'][t][h] = C * self.qc1  # 非控制区域3的需求，直接通过到达率计算
           else:
               self.i_mc['出口2'][t][h] = self.o_m['出口1'][cc3][h - 1] * self.alpha[2] + self.o_m['出口2'][cc3][h - 1] * self.alpha[1] + \
                                   self.o_m['出口3'][cc3][h - 1] * self.alpha[0]  # 存疑 ，
           if h == 3:
               self.i_mc['出口4'][t][h] = C * self.qc2  # 非控制区域4的需求，直接通过到达率计算
           else:
               self.i_mc['出口4'][t][h] = self.o_mc['出口1'][cc4][h + 1] * self.alpha[0] + self.o_mc['出口3'][cc4][h + 1] * self.alpha[2] + \
                                   self.o_mc['出口4'][cc4][h + 1] * self.alpha[1]



       if np.all((self.n_m['出口1'][t][h] * self.alpha[0] + self.n_m['出口3'][t][h] * self.alpha[0]) < action[h] * self.sta_flow * 2):
           self.l_mc['出口1'][t][h] = self.n_mc['出口1'][t][h] * self.alpha[0]  # z这里应该是n_mc
           self.l_mc['出口3'][t][h] = self.n_m['出口3'][t][h] * self.alpha[0]  # 这里不区分公交车与社会车么？？
           self.d_mlc['出口1'][t + 1][h] = 0
           self.d_mlc['出口3'][t + 1][h] = 0
       else:
           l_mb['出口1'][t][h] = round((n_mb['出口1'][t][h] / n_m['出口1'][t][h]) * g_l[h] * sta_flow)  # 等待车辆数中的公交车与社会车占比进行排放
           l_mc['出口1'][t][h] = round((n_mc['出口1'][t][h] / n_m['出口1'][t][h]) * g_l[h] * sta_flow)
           l_m['出口1'][t][h] = l_mb['出口1'][t][h] + l_mc['出口1'][t][h]
           l_mb['出口3'][t][h] = round((n_mb['出口3'][t][h] / n_m['出口3'][t][h]) * g_l[h] * sta_flow)
           l_mc['出口3'][t][h] = round((n_mc['出口3'][t][h] / n_m['出口3'][t][h]) * g_l[h] * sta_flow)
           l_m['出口3'][t][h] = l_mb['出口3'][t][h] + l_mc['出口3'][t][h]
           d_mlb['出口1'][t + 1][h] = math.ceil(
               (alpha[0] * n_mb['出口1'][t][h] - l_mb['出口1'][t][h]) / g_l[h] * sta_flow) * C * (
                                                alpha[0] * n_mb['出口1'][t][h] - l_mb['出口1'][t][h]) * num_b
           d_mlc['出口1'][t + 1][h] = math.ceil(
               (alpha[0] * n_mc['出口1'][t][h] - l_mc['出口1'][t][h]) / g_l[h] * sta_flow) * C * (
                                                alpha[0] * n_mc['出口1'][t][h] - l_mc['出口1'][t][h]) * num_c
           d_mlb['出口3'][t + 1][h] = math.ceil(
               (alpha[0] * n_mb['出口3'][t][h] - l_mb['出口3'][t][h]) / g_l[h] * sta_flow) * C * (
                                                alpha[0] * n_mb['出口3'][t][h] - l_mb['出口3'][t][h]) * num_b
           d_mlc['出口3'][t + 1][h] = math.ceil(
               (alpha[0] * n_mc['出口3'][t][h] - l_mb['出口3'][t][h]) / g_l[h] * sta_flow) * C * (
                                                alpha[0] * n_mc['出口3'][t][h] - l_mc['出口3'][t][h]) * num_c






# 定义Net类 (定义网络)
class Net(nn.Module):
    def __init__(self):                                                         # 定义Net的一系列属性
        # nn.Module的子类函数必须在构造函数中执行父类的构造函数
        super(Net, self).__init__()                                             # 等价与nn.Module.__init__()

        self.fc1 = nn.Linear(N_STATES, 50)                                      # 设置第一个全连接层(输入层到隐藏层): 状态数个神经元到50个神经元
        self.fc1.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)
        self.out = nn.Linear(50, N_ACTIONS)                                     # 设置第二个全连接层(隐藏层到输出层): 50个神经元到动作数个神经元
        self.out.weight.data.normal_(0, 0.1)                                    # 权重初始化 (均值为0，方差为0.1的正态分布)

    def forward(self, x):                                                       # 定义forward函数 (x为状态)
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
            action = action[0]                                                  # 输出action的第一个数
        else:                                                                   # 随机选择动作
            action = np.random.randint(0, N_ACTIONS)                            # 这里action随机等于0或1 (N_ACTIONS = 2)
        return action                                                           # 返回选择的动作 (0或1)

    def store_transition(self, s, a, r, s_):                                    # 定义记忆存储函数 (这里输入为一个transition)
        transition = np.hstack((s, [a, r], s_))                                 # 在水平方向上拼接数组
        # 如果记忆库满了，便覆盖旧的数据
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


for i in range(400):                                                    # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()                                                     # 重置环境
    episode_reward_sum = 0                                              # 初始化该循环对应的episode的总奖励

    while True:                                                         # 开始一个episode (每一个循环代表一步)
        env.render()                                                    # 显示实验动画
        a = dqn.choose_action(s)                                        # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)                                 # 执行动作，获得反馈

        # 修改奖励 (不修改也可以，修改奖励只是为了更快地得到训练好的摆杆)
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        new_r = r1 + r2

        dqn.store_transition(s, a, new_r, s_)                 # 存储样本
        episode_reward_sum += new_r                           # 逐步加上一个episode内每个step的reward

        s = s_                                                # 更新状态

        if dqn.memory_counter > MEMORY_CAPACITY:              # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            dqn.learn()
        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break




class env:#输入action,输出下一个状态值，奖赏值，是否结束，其他信息
    def action(self):


    l1 = [3.1, 2.9, 3.0, 3.3]  # 四个路口的长度
    l2 = [1.9, 2.1, 2.0, 1.3]
    lh = [1.2, 1.35, 1.5]  # 交叉口之间的距离，单位 km
    alpha = [0.3, 0.5, 0.2]  # 0.3左转，0.5直行，0.2右转
    ps = 80
    lv = 4  # 私家车长度 ， 单位m
    n_m = {1: np.zeros((900, 4)), 2: np.zeros((900, 4)), 3: np.zeros((900, 4)),
           4: np.zeros((900, 4))}  # 交叉口进口等待的社会车与公交车之和

    td_c = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
            '出口4': np.zeros((900, 4))}  # 私家车时延

    i_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口进口的社会车与公交车之和

    vb = 11  # 公交速度 单位m/s
    h_list = []
    # r1 = np.zeros((900,4))           # r1是区域1社会车诱导参数，通过logit模型计算，这里为了试验直接给定值
    # r2 = np.zeros((900,4))           # r2是区域2社会车诱导参数，通过logit模型计算，这里为了试验直接给定值
    # r = 0.25
    r1 = np.zeros((900, 4))
    r2 = np.zeros((900, 4))
    sta_flow = 0.8  # 饱和流
    qb1 = 0.01
    qc1 = 0.05
    qb2 = 0.01
    qc2 = 0.05

    l_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 四个进口道的左转车辆数

    s_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}

    o_m = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)),
           '出口4': np.zeros((900, 4))}  # 交叉口出口的社会车与公交车之和

    o = []
    obj = []

    # 非控制区域到达交叉口的到达率
    # 人均等待时间
    d_mlc = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}

    d_msc = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
    d_hb = np.zeros((900, 4))
    d_hc = np.zeros((900, 4))
    d_hm = {'出口1': np.zeros((900, 4)), '出口2': np.zeros((900, 4)), '出口3': np.zeros((900, 4)), '出口4': np.zeros((900, 4))}
    # d_h = np.zeros((900,4))
    num_b = 20
    num_c = 1.5

    beta1 = [0.25, 0.25, 0.25, 0.25]
    beta2 = [0.25, 0.25, 0.25, 0.25]

    nnnnn = []



from .demand import d
# from GA import vc
#############################  second_layer 参数 #############################################

########################################################################################
######################################################### second_layer 参数 ###########################################
def init():
    pq = 160#种群数量
    f = 0.5#缩放因子
    cr = 0.8# 交叉概率
    gen = 1000# 遗传代数
    len_g = 8
    g_min = 20
    g_max = 60
    return pq, f, cr, gen, len_g, g_min, g_max


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
            i_mc['出口1'][t][h] = C * (1/4 * d[1_2][cc1])  # 社会车通过logit模型获得分流率r1，获得进入各个路口的车辆数
            i_mc['出口3'][t][h] = C * (d[2_1][cc2]/4)
            if h == 0:
                i_mc['出口2'][t][h] = C * qc1  # 非控制区域3的需求，直接通过到达率计算
            else:
                i_mc['出口2'][t][h] = o_mc['出口1'][cc3][h - 1] * alpha[2] + o_mc['出口2'][cc3][h - 1] * alpha[1] + \
                                    o_mc['出口3'][cc3][h - 1] * alpha[0]  # 存疑 ，
            if h == 3:
                i_mc['出口4'][t][h] = C * qc2  # 非控制区域4的需求，直接通过到达率计算
            else:
                i_mc['出口4'][t][h] = o_mc['出口1'][cc4][h + 1] * alpha[0] + o_mc['出口3'][cc4][h + 1] * alpha[2] + \
                                    o_mc['出口4'][cc4][h + 1] * alpha[1]

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
    for i in n_m:
            nm11 += n_m[i][t][0]
            nm22 += n_m[i][t][1]
            nm33 += n_m[i][t][2]
            nm44 += n_m[i][t][3]  # 计算各交叉口等待车辆数
    n1111.append(nm11)
    n2222.append(nm22)
    n3333.append(nm33)
    n4444.append(nm44)
    min_num = abs(com1 - emi1)+abs(com2 - emi2)
    return min_num,d_h





########################################## main ############################################
if __name__ == "__main__":
    min_obj = []
    min_g_l=[]
    min_g_s=[]
    gl=[]
    gs=[]
    min_fff=[]
    d_hh=[]
    pq, f, cr, gen, len_g, g_min, g_max = init()
pq1_list, pq2_list = initialtion2(pq)
def result2():
    pq, f, cr, gen, len_g, g_min, g_max = init()#初始化
    pq1_list, pq2_list = initialtion2(pq)#初始化种群
    for i in range(0, pq):
        x = []
        x.append(objective2(pq1_list[i], pq2_list[i]))#对每个个体进行评价，并存放在x中
    min_obj.append(min(x))#找出最好目标函数的值，存放在min_obj
    min_g_l.append(pq1_list[x.index(min(x))])#找出最好的个体存放在min_gl
    min_g_s.append(pq2_list[x.index(min(x))])
    #gen代表繁衍次数=1000
    for i in range(gen):
        v1_list, v2_list = mutation2(pq1_list, pq2_list)
        u1_list, u2_list = crossover2(pq1_list, pq2_list, v1_list, v2_list)
        pq1_list, pq2_list = selection2(u1_list, u2_list, pq1_list, pq2_list)
        for j in range(pq):
            xx = []
            xx.append(objective2(pq1_list[j], pq2_list[j]))
        min_obj.append(min(xx))
        min_g_l.append(pq1_list[xx.index(min(xx))])
        min_g_s.append(pq2_list[xx.index(min(xx))])
    min_f = min(min_obj)
    min_g_ll = min_g_l[min_obj.index(min_f)]
    min_g_ss = min_g_s[min_obj.index(min_f)]
    gl.append(min_g_ll)
    gs.append(min_g_ss)
    min_fff.append(min_f)
    min_num2 ,d_h2 = objective2(min_g_ll, min_g_ss)
    # h1n.append(d_hb)
    # h2n.append(d_hc)
    d_hh.append(d_h2)




#     pq = 10
# f = 0.5
# cr = 0.8
# gen = 1000
# len_g = 8
# g_min = 20
# g_max = 60
# pq1_list, pq2_list = initialtion2(pq)#形成初始种群，种群大小为pq=10
# print(pq1_list)
# print(pq2_list)
# objective2(pq1_list[0], pq2_list[0])
