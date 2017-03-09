#K-fold 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import kaggle
errors={}
e=[]

def kFold(features_train,labels_train,features_test,labels_test,path,DataSet):
    n=features_train.shape
    k=10 # make it 10 fold test
    size = n[0]/k # size of each fold.
    errors1={} # to save CrossValidation Errors
    errors2={} # to save Training Errors
    e1=[]      # to obtain mean square error of whole CrossValidation set
    e2=[]      #to obtain mean square error of whole Training set
    for p in range(1,20):          # considering neighbours from range 1 to 20.
        print "Considering Neighbor="+str(p)
        for i in range(1,k): 
            # Select the cross Validation set, it will change with changing values of i       
            Feature_CrossVal = features_train[i*size:][:size]  
            Label_CrossVal = labels_train[i*size:][:size]   
            
            # Add the rest of Training set  
            Feature_Train = features_train[:i*size]
            np.append(Feature_Train ,features_train[(i+1)*size:])  
            Label_Train = labels_train[:i*size]
            np.append(Label_Train ,labels_train[(i+1)*size:])
            
            # K Nearest Neighbor training with testset and testing with Cross Validation set
            neigh = KNeighborsClassifier(n_neighbors=p)
            neigh.fit(Feature_Train,Label_Train) 
            predict=np.zeros(Label_CrossVal.shape)
            predict=neigh.predict(Feature_CrossVal)
            
            # K Nearest Neighbor training and testing on same set
            ne = KNeighborsClassifier(n_neighbors=p)
            ne.fit(Feature_Train,Label_Train) 
            pre=np.zeros(Label_Train.shape)
            pre=ne.predict(Feature_Train)
            
            # Find the mean square error between the found predictions to the cross validation output set add error in a list 
            e1.append(mean_squared_error(Label_CrossVal,predict))   
            e2.append(mean_squared_error(Label_Train,pre))    
            print e1,e2 # printing so that it doesn't look like program got hanged :P
            
        # take mean of the K fold errors for particular neighbor and append it   
        errors1[p]=np.mean(e1)   
        errors2[p]=np.mean(e2)

    #Find predictions with Min Error Neighbor
    neigh = KNeighborsClassifier(n_neighbors=min(errors1, key=errors1.get))
    #fitting the training data
    neigh.fit(features_train,labels_train) 
    predict=np.zeros(labels_test.shape)
    #testing with test features
    predict=neigh.predict(features_test)
    kaggle.kaggleize(predict, str(path)+"/Submission/Predictions/"+str(DataSet)+"/CrossValidation_KNN.csv")
    #returning errors to run_me.py file for plotting CrossValidation and Training Error
    
    return errors1,errors2

