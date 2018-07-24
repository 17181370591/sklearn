import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

#from sklearn.cross_validation import cross_val_score     #这个将被移除，用下面那行
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt


iris=datasets.load_iris()
iris_X=iris.data
iris_Y=iris['target']

X_train,X_test,Y_train,Y_test=train_test_split(iris_X,iris_Y,test_size=.8,
                                            #random_state=1,
                                            ）


knn=KNeighborsClassifier(n_neighbors=5)                   #knn用来预测新数据属于哪一类
#进行cv次test，每次用1/cv个数据作test，cv次能test所有的数据X_test
scores=cross_val_score(knn,X_test,Y_test,cv=6,scoring='accuracy')       
print(scores)
print(scores.mean())



krange=range(1,31)
kscore=[]
for k in krange:
    knn=KNeighborsClassifier(n_neighbors=k)                   #knn用来预测新数据属于哪一类
    scores=cross_val_score(knn,X_test,Y_test,cv=6,scoring='accuracy')
    kscore.append(scores.mean())

#画图查看哪个k参数最好
plt.plot(krange,kscore)
plt.show()
