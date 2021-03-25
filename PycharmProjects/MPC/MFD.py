import matplotlib.pyplot as plt
## MFD模型参数
N1 = 100
n11 = 80
n12 = 50
N2 = 160
n21 = 50
n22 = 40
N1_C = 2100
N2_C = 1300
N1_JAM = 4000
N2_JAM = 2500
l = [31.04, 21.415]
l1 =[3.1,2.9,3.0,3.3]
l2 =[1.9,2.1,2.0,1.3]
t_p = 20  # 预测步数
t_c = 360  # 控制时间 单位为秒
C = 120
t_s = 300  # 控制步数
x=[]
y=[]
M2=[]
M1=[]
G1 = -1E-07* pow(N1, 2) + 4E-04 *N1 + 0.022
G2 = -2E-07*pow(N2, 2) + 5E-04 *N2 + 0.0478
for N in range(4000):
    x.append(N)
    G1 = -1E-07* pow(N, 2) + 4E-04 *N + 0.022
    M1.append(G1)
for N in range(2500):
    y.append(N)
    G2 = -2E-07*pow(N, 2) + 5E-04 *N + 0.0478
    M2.append(G2)



plt.figure(figsize=(6,4))
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.xlabel('区域累计车辆数')
plt.ylabel('完成流（veh/sec)')
plt.plot(y,M2,label='M2_1',linewidth=1)
plt.plot(x,M1,label='M1_2',linewidth=0.5)
plt.legend(loc='upper left')
plt.title('实际转移值')
plt.grid(True)
plt.show()
