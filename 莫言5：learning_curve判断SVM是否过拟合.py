#https://blog.csdn.net/zhouwenyuan1015/article/details/65448285

import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import learning_curve
from sklearn.svm import SVC



digits=datasets.load_digits()
X=digits.data
Y=digits['target']                  #1797



'''
#cv=3表示每个数据测试5次，每个数据训练数据和测试数据比例是4:1，会收集到5个数据，
#而train_sizes表示记录6次，所以train_loss,test_loss是6*5；
#似乎learning_curve里，train_sizes的每个值a表示训练了百分比a的值时，
#记录一次训练样本数，训练集上准确率,交叉验证集上的准确率
#修改gamma可能会发现图像先降后升，这说明测试的数据越多，进行测试时效果反而变差，原因是过拟合
'''

train_sizes,train_loss,test_loss=learning_curve(
        SVC(gamma=.01),X,Y,cv=10,scoring='neg_mean_squared_error',train_sizes=np.arange(.1,1.1,.1))
train_loss_mean=-np.mean(train_loss,axis=1)
test_loss_mean=-np.mean(test_loss,axis=1)



plt.plot(train_sizes,train_loss_mean,'o-',color='r',label='train')
plt.plot(train_sizes,test_loss_mean,'o-',color='g',label='test')
plt.xlabel('training examples')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()
