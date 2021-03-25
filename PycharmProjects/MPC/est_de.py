import random
import matplotlib.pyplot as plt
import pickle

a, b, c = 60, 180, 240
p, q = 0.3, 0.17
x,y,z = 60, 180, 240#高需求
m,n = 0.33, 0.2#高需求

class TrafficDemand:
    def __init__(self):
        self.d = {1_1: [], 1_2: [], 2_1: [], 2_2: []}
        self.d_low = {1_1: [], 1_2: [], 2_1: [], 2_2: []}
        self.x = []
        self.y = []
        self.d_2 = self.highdemand(x,y,z,m,n)
        self.d_low_2 = self.lowdemand(a,b,c,p,q)

    ############## 产生高需求，控制300步，一步三次需求 #############
    def highdemand(self,x,y,z,m,n):
        for i in range(300):
            if i <= x:
                self.x.append(i)
                a1 = random.uniform(0, 0.05) + (m / x) * i
                a2 = random.uniform(0, 0.05) + (m / x) * i
                a3 = random.uniform(0, 0.01) + (n / x) * i
                a4 = random.uniform(0, 0.01) + (n / x) * i
                self.d[1_1].append(round(a1, 4))
                self.d[1_2].append(round(a2, 4))
                self.d[2_1].append(round(a3, 4))
                self.d[2_2].append(round(a4, 4))
            elif x < i <= y:
                self.x.append(i)
                a1 = random.uniform(m, m+0.05)
                a2 = random.uniform(m, m+0.05)
                a3 = random.uniform(n, n+0.05)
                a4 = random.uniform(n, n+0.05)
                self.d[1_1].append(round(a1, 4))
                self.d[1_2].append(round(a2, 4))
                self.d[2_1].append(round(a3, 4))
                self.d[2_2].append(round(a4, 4))
            elif y < i <= z:
                self.x.append(i)
                a1 = random.uniform(m*z/(z-y), m*z/(z-y)) + (-m / (z-y)) * i
                a2 = random.uniform(m*z/(z-y), m*z/(z-y)) + (-m / (z-y)) * i
                a3 = random.uniform(n*z/(z-y), n*z/(z-y)) + (-n / (z-y)) * i
                a4 = random.uniform(n*z/(z-y), n*z/(z-y)) + (-n / (z-y)) * i
                self.d[1_1].append(round(a1, 4))
                self.d[1_2].append(round(a2, 4))
                self.d[2_1].append(round(a3, 4))
                self.d[2_2].append(round(a4, 4))
            else:
                self.x.append(i)
                self.d[1_1].append(0)
                self.d[1_2].append(0)
                self.d[2_1].append(0)
                self.d[2_2].append(0)
        return self.d

    def fig_highdemand(self):
        plt.figure(figsize=(6, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.xlabel('time(2min)')
        plt.ylabel('demand(veh/sec)')
        plt.plot(self.x, self.d[1_1], label='d11', linewidth=0.5)
        plt.plot(self.x, self.d[1_2], label='d12', linewidth=0.5)
        plt.plot(self.x, self.d[2_1], label='d21', linewidth=0.5)
        plt.plot(self.x, self.d[2_2], label='d22', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.title('高需求')
        plt.show()
        # print(d)

    ############## 产生低需求，控制300步，一步三次需求 #############

    def lowdemand(self,a,b,c,p,q):
        for i in range(300):
            if i <= a:
                self.y.append(i)
                a1 = random.uniform(0, 0.05) + (p / a) * i
                a2 = random.uniform(0, 0.05) + (p / a) * i
                a3 = random.uniform(0, 0.01) + (q / a) * i
                a4 = random.uniform(0, 0.01) + (q / a) * i
                self.d_low[1_1].append((round(a1, 4)))
                self.d_low[1_2].append((round(a2, 4)))
                self.d_low[2_1].append((round(a3, 4)))
                self.d_low[2_2].append((round(a4, 4)))
            elif a < i <= b:
                self.y.append(i)
                a1 = random.uniform(p, p+0.05)
                a2 = random.uniform(p, p+0.05)
                a3 = random.uniform(q, q+0.05)
                a4 = random.uniform(q, q+0.05)
                self.d_low[1_1].append((round(a1, 4)))
                self.d_low[1_2].append((round(a2, 4)))
                self.d_low[2_1].append((round(a3, 4)))
                self.d_low[2_2].append((round(a4, 4)))
            elif b < i <= c:
                self.y.append(i)
                a1 = random.uniform(p*c/(c-b), p*c/(c-b)) + (-p / (c-b)) * i
                a2 = random.uniform(p*c/(c-b), p*c/(c-b)) + (-p / (c-b)) * i
                a3 = random.uniform(q*c/(c-b), q*c/(c-b)) + (-q / (c-b)) * i
                a4 = random.uniform(q*c/(c-b), q*c/(c-b)) + (-q / (c-b)) * i
                self.d_low[1_1].append((round(a1, 4)))
                self.d_low[1_2].append((round(a2, 4)))
                self.d_low[2_1].append((round(a3, 4)))
                self.d_low[2_2].append((round(a4, 4)))
            else:
                self.y.append(i)
                self.d_low[1_1].append(0)
                self.d_low[1_2].append(0)
                self.d_low[2_1].append(0)
                self.d_low[2_2].append(0)
        return self.d_low

    def fig_lowdemand(self):
        plt.figure(figsize=(6, 4))
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.xlabel('time(2min)')
        plt.ylabel('demand(veh/sec)')
        plt.plot(self.y, self.d_low[1_1], label='d11', linewidth=0.5)
        plt.plot(self.y, self.d_low[1_2], label='d12', linewidth=0.5)
        plt.plot(self.y, self.d_low[2_1], label='d21', linewidth=0.5)
        plt.plot(self.y, self.d_low[2_2], label='d22', linewidth=0.5)
        plt.legend(loc='upper left')
        plt.title('低需求')
        plt.show()
        # print(d)

aa = TrafficDemand()
# aa.fig_highdemand()
f = open(r'E:\wyy\PycharmProjects\MPC\temp_file\aa.pkl', 'wb')
pickle.dump(aa, f, 0)
f.close()





