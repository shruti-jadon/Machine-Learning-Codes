import kaggle

# Assuming you are running run_me.py from the Submission/Code
# directory, otherwise the path variable will be different for you
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn import linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline

def Experiment_DataSet(features_train,labels_train,features_test,labels_test,path,i): 
      
    #Support Vector Regression
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    y_rbf=svr_rbf.fit(features_train, labels_train).predict(features_test)
    kaggle.kaggleize(y_rbf, path+"Submission/Predictions/"+str(i)+"/RFE_SVR.csv")
    print "Support Vector Regression"
    
    #K Nearest Neighbor
    neigh = KNeighborsRegressor(n_neighbors=6)
    neigh.fit(features_train, labels_train)
    predict=neigh.predict(features_test)
    kaggle.kaggleize(predict, path+"Submission/Predictions/"+str(i)+"/KNN_REG.csv")
    print "K Nearest Neighbor"
    
    #LassoCV with RFE and pipeline merge of both
    alpha=np.arange(0.1,2,0.1) #defining range of alphas 
    lasso = linear_model.LassoCV(alphas=alpha)
    rfe = RFE(estimator=lasso, step=1)
    Lasso_Pipeline = make_pipeline(rfe, svr_rbf)
    Lasso_Pipeline.fit(features_train, labels_train)
    predict=Lasso_Pipeline.predict(features_test)
    kaggle.kaggleize(predict, path+"Submission/Predictions/"+str(i)+"/LassoCV.csv")
    print "LassoCV with RFE and pipeline"
    
    

