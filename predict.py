import pandas as pd
import tensorflow as tf
import numpy as np
from readData import data_test,specArr
from tensorflow.keras.models import load_model

#加载模型
model = load_model('./model/save_model.h5')

#4335     260  150    560  450
#3948     270  120    570  420
#4034     250     150    550   450
#4442    311     240      611    540
# data_test=data_test.iloc[:,1:-1]
# data_test = np.expand_dims(data_test,-1)
# data_test=data_test.reshape(data_test.shape[0],16,16,1)
#
#
# res=model.predict(data_test)
import turtle as t

t.speed(10)

t.screensize(canvwidth=696, canvheight=682, bg='white')

you_list=[]

for ii in range(207,507):
    for jj in range(118,518):
        x=specArr[ii][jj]
        x = pd.DataFrame([x])
        x = np.expand_dims(x,-1)
        x=x.reshape(x.shape[0],256,1)
        data=model.predict(x)
        if data[0][1]>0.5:
            you_list.append([ii,jj])
            print([ii,jj])
            t.tracer(0)
            t.color('black', 'black')
            t.penup()
            t.goto(ii - 341, -jj + 348)
            t.pendown()
            t.dot(2)
            t.update()
t.done()
pd.DataFrame(you_list).to_excel('./data/4525.xlsx')




# for ii in range(198,696):
#     for jj in range(437,682):
#         # specArr[ii][jj]
#         x=specArr[ii][jj]
#         x=pd.DataFrame([x])
#         x = np.expand_dims(x,-1)
#         x=x.reshape(x.shape[0],256,1)
#         data=model.predict(x)
#         print(data)
#         if data[0][1]>0.5:
#             t.tracer(0)
#             t.color('black', 'black')
#             t.penup()
#             t.goto(ii - 341, -jj + 348)
#             t.pendown()
#             t.dot(2)
#             t.update()
#         else:
#             t.tracer(0)
#             t.color('red', 'red')
#             t.penup()
#             t.goto(ii - 341, -jj + 348)
#             t.pendown()
#             t.dot(2)
#             t.update()





# print(res)

