import pickle

f = open(r'E:\\wyy\\PycharmProjects\\MPC\\temp_file\\aa.pkl', 'rb')
bb = pickle.load(f)
f.close()
d=bb.d_2
d_low = bb.d_low_2
print(d)
print(d_low)







