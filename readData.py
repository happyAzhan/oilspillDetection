import pandas as pd
import numpy as  np
path1 = '.\\data\\xlsx\\纯油点.xlsx'
path2 = '.\\data\\xlsx\\水点.xlsx'

data_test1='.\\data\\xlsx\\水测试.xlsx'
data_test2='.\\data\\xlsx\\油测试.xlsx'

raw1 = r".\\data\\raw\\newrawfile20211104174525.raw"
# 读取训练集
# 数据1
data1 = pd.read_excel(path1,header=None)
data1 = data1[~data1[0].isin(['X','Index'])]

data2 = pd.read_excel(path2,header=None)
data2 = data2[~data2[0].isin(['X','Index'])]



# 读取测试集
# 测试数据

data_test1 = pd.read_excel(data_test1,header=None)
data_test1=data_test1[~data_test1[0].isin(['X','Index'])]

data_test2 = pd.read_excel(data_test2,header=None)
data_test2=data_test2[~data_test2[0].isin(['X','Index'])]

data_test=pd.concat([data_test1,data_test2])
data_test=data_test1
data = pd.concat([data1,data2,data_test1])
#读取raw原始数据 #682*696
f = open(raw1,'rb')
fint = np.fromfile(f,dtype = np.int16)
specArr=[]

for ii in range(696):
    dim2=[]
    for jj in range(682):
        dim3 = []
        for zz in range(256):
            dim3.append(fint[ii + (178176 * jj) + (696 * zz)])
        dim2.append(dim3)
    specArr.append(dim2)
# print(specArr[233][12])