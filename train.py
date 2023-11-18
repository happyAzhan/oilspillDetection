import tensorflow as tf
from keras.optimizer_v2.rmsprop import RMSprop
import pandas as pd
from readData import data,data_test
import numpy as np
from tensorflow.keras.layers import Flatten,Conv2D,Dropout,Input,Dense,MaxPooling2D
from tensorflow.keras.models import Model
data=data.sample(frac=1).reset_index(drop=True)
data_train=data.iloc[0:200,:]
data=data.iloc[200:,:]
data_test=data_test.sample(frac=1).reset_index(drop=True)

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
x_train=data.iloc[:,1:-1]
y_train=data.iloc[:,-1]

x_test=data_train.iloc[:,1:-1]
y_test=data_train.iloc[:,-1]


# model=tf.keras.Sequential()
# model.add(tf.keras.layers.Dense(256,input_shape=(256,),activation='relu'))
# model.add(tf.keras.layers.Dense(128,activation='relu'))
# model.add(tf.keras.layers.Dropout(0.1))
# model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
# model.summary()
# model.compile(
#     optimizer='adam',
#     loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
#     metrics=['acc']
# )
# history=model.fit(x_train,y_train,epochs=250,batch_size=32)





x_train = np.expand_dims(x_train,-1)
x_train=x_train.reshape(x_train.shape[0],256,1)

x_test = np.expand_dims(x_test,-1)
x_test=x_test.reshape(x_test.shape[0],256,1)

# 作为输入
inputs = Input([16,16,1])
# 搭建网络结构
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(256, 3, activation='relu', input_shape=(256,1)),
    tf.keras.layers.Conv1D(64,  3, activation='relu'),
    tf.keras.layers.MaxPooling1D( 2),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(2, activation='softmax')
])
# 编译网络模型
model.compile(optimizer=RMSprop(lr=0.000001),  loss='sparse_categorical_crossentropy', metrics=['acc'])





# 利用fit进行训练
history=model.fit(x_train, y_train, epochs=150,validation_data=(x_test, y_test))






import matplotlib.pyplot as plt
print('--------------------------------------')
print(history.history.keys())
print('--------------------------------------')
plt.plot(history.epoch,history.history.get('loss'))
plt.plot(history.epoch,history.history.get('val_loss'))
loss=history.history.get('loss')
val_loss=history.history.get('val_loss')
acc=history.history.get('acc')
val_acc=history.history.get('val_acc')
pd.DataFrame(loss).to_excel('./data/1Dloss.xlsx')
pd.DataFrame(val_loss).to_excel('./data/1Dval_loss.xlsx')
pd.DataFrame(acc).to_excel('./data/1Dacc.xlsx')
pd.DataFrame(val_acc).to_excel('./data/1Dval_acc.xlsx')
plt.show()
model.save('./model/save_model.h5')
