#https://www.bilibili.com/video/av17003173/?p=5
#knn用来预测新数据属于哪一类

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
iris_X=iris.data
iris_Y=iris.target
#iris_Y=iris['target']

#80%的数据用来训练，设置random_state可以使每次的数据都一样，从而去掉每次随机得到数据不一样使结果产生影响
X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,random_state=4,test_size=.8)
print(iris_Y,Y_test)              #这里可以发现iris_Y数据是排好序的而Y_test是乱序的

knn=KNeighborsClassifier(n_neighbors=5)       #knn用来预测新数据属于哪一类，这里取最近的5的邻居决定数据的分类
knn.fit(X_train,Y_train)

res=knn.predict(X_test)

print(knn.score(X_test,Y_test))
