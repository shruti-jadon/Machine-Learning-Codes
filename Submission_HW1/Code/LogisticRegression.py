from sklearn import linear_model
import time
import numpy as np

def LR(features_train,labels_train,features_test,labels_test):
    Train_time = time.time()
    logreg = linear_model.LogisticRegression(C=1.0)
    logreg.fit(features_train, labels_train)
    Train_time=time.time()-Train_time   
    #LR Testing
    Test_time = time.time()
    predict=np.zeros(labels_test.shape)
    predict=logreg.predict(features_test)
    Test_time=time.time()-Test_time
    return predict,Train_time, Test_time

