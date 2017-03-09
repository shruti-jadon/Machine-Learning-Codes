#test accuracy, training time, and prediction time
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import time

def supportvector(features_train,labels_train,features_test,labels_test):
    Train_time = time.time()
    Svm_Sklearn = svm.SVC()
    Svm_Sklearn.fit(features_train, labels_train) 
    Train_time=time.time()-Train_time
    Test_time=time.time()
    predict=np.zeros(labels_test.shape)
    predict=Svm_Sklearn.predict(features_test)
    Test_time=time.time()-Test_time
    return predict,Train_time, Test_time



