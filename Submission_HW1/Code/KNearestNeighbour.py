#K Nearest Neighbor
import time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
 
def KNN(features_train,labels_train,features_test,labels_test):
    Train_time = time.time()
    Nearest_neighbours = KNeighborsClassifier(n_neighbors=5) # Defaultk=5
    Nearest_neighbours.fit(features_train, labels_train)   
    Train_time=time.time()-Train_time
    Test_time=time.time()
    predict=np.zeros(labels_test.shape)
    predict=Nearest_neighbours.predict(features_test)
    Test_time=time.time()-Test_time
    return predict,Train_time, Test_time