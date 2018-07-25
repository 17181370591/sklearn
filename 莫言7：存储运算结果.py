
import numpy as np,pickle,time
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split,cross_val_score
import matplotlib.pyplot as plt
from sklearn.learning_curve import validation_curve
from sklearn.svm import SVC
from sklearn.externals import joblib



digits=datasets.load_digits()
X=digits.data
Y=digits['target']                  #1797
clf=SVC()
clf.fit(X,Y)



'''
#方法1：
with open('clf.pickle','wb') as f:
    pickle.dump(clf,f)
time.sleep(1)
with open('clf.pickle','rb') as f:
    clf2=pickle.load(f)
    print(clf2.score(X,Y))
'''


#方法2：
joblib.dump(clf,'clf.pkl')
clf3=joblib.load('clf.pkl')
print(clf3.score(X,Y))
