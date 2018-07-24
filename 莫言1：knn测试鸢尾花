#https://www.bilibili.com/video/av17003173/?p=5

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
iris_X=iris.data
iris_Y=iris.target
#iris_Y=iris['target']

#80%的数据用来训练
X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size=.8)
print(iris_Y,Y_test)              #这里可以发现iris_Y数据是排好序的而Y_test是乱序的

knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)

res=knn.predict(X_test)

no=np.argwhere(res!=Y_test)
print(no)
print(1-no.size/res.size)
