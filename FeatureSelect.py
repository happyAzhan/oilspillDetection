import pandas as pd
# 首先导入数据，查看数据集的基本情况：
from sklearn.ensemble import RandomForestClassifier
from readData import data
import numpy as np

df = data
x=df[df.columns[1:257]]
y=df[df.columns[257]]
df.head()
df.shape


# df['loan_status'].unique()
# df['y'] = df['loan_status'].map(lambda x: int((x == 'Charged Off') | (x == 'Late (31-120 days')))
# df.drop('loan_status', axis=1,inplace=True)


#训练模型，这里随机森林模型参数都用默认值
# y = df['y']
# x = df.drop('y', axis=1)
clf = RandomForestClassifier()
clf.fit(x, y)

importance = clf.feature_importances_

indices = np.argsort(importance)[::-1]
features = x.columns

for f in range(x.shape[1]):
	print(("%f" % ( importance[f])))