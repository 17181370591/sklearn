#https://www.bilibili.com/video/av17003173/?p=6
#线性回归生成一个拟合函数，来预测新数据的值

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

loaded_data=datasets.load_boston()      #用波士顿房价测试线性回归，生成一个拟合函数来预测新数据的值
data_X=loaded_data.data
data_Y=loaded_data.target

model=LinearRegression()
model.fit(data_X,data_Y)

res=model.predict(data_X)

print(np.absolute((data_Y-res)/data_Y))

x,y=datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=5)      #生成随机数据
model=LinearRegression()
model.fit(x,y)
res=model.predict(x)

plt.scatter(x,y,color='r')
plt.scatter(x,res)
plt.show()
