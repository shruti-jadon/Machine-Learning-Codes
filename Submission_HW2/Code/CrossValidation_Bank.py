import numpy as np
import kaggle
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
 
def CrossValidation_Bank(features_train,labels_train,features_test,labels_test,path):
    n=features_train.shape #determine shape of training set features
    k=2 #Passing k=2. i.e; 2 fold cross validation
    size = n[0]/k # size of each fold.
    errors1={} # defining a dictionary to save the Hyperparameter along with obtained errors
    e1=[] #to store the errors while cross validation loop
    alphavalues=np.arange(1,10,1) #defining range of difference in hyper-parameter
    for p in alphavalues: # considering neighbours from range 1 to 20.
        print "Considering"+str(p)
        alpha=np.arange(0.1+p, 2+p, 0.1) # varying the range of alpha with change in p
        for i in range(1,k): 
            # Select the cross Validation set, it will change with changing values of i       
            Feature_CrossVal = features_train[i*size:][:size]  
            Label_CrossVal = labels_train[i*size:][:size]   
            
            # Add the rest of Training set  
            Feature_Train = features_train[:i*size]
            np.append(Feature_Train ,features_train[(i+1)*size:])  
            Label_Train = labels_train[:i*size]
            np.append(Label_Train ,labels_train[(i+1)*size:])
            
            #Defining Ridge Regression and passing range of values of alpha(Regularization Constant)
            svr_rbf = RidgeCV(alphas=alpha)
            #Defining Recursive Feature Eliminiation with step=1, i.e; only 1 feature gets eliminated at 1 step.
            rfe = RFE(estimator=svr_rbf, step=1)
            # Defining pipeline with passing Ridge and RFE as parameter.
            anova_svm = make_pipeline(rfe, svr_rbf)
            anova_svm.fit(Feature_Train, Label_Train)
            #defining array of zeros of label shape, to save the predictions
            predict=np.zeros(labels_test.shape)
            predict=anova_svm.predict(Feature_CrossVal)
            # adding the obtained absolute error in e1 for this fold cycle.
            e1.append(np.sqrt(mean_squared_error(Label_CrossVal,predict)))      
            print e1 
            # take mean of the K fold errors for particular neighbor and append it   
            errors1[p]=np.mean(e1) 
    
    print min(errors1, key=errors1.get) # print the minimum error hyper-parameter range
    alpha=np.arange(0.1+min(errors1, key=errors1.get), 2+min(errors1, key=errors1.get), 0.1) #define alpha range with min obtained range value.
    svr_rbf = RidgeCV(alphas=alpha) #define Ridge Regression
    rfe = RFE(estimator=svr_rbf, n_features_to_select=n[1], step=1) #define Recursive Feature Elimination
    anova_svm = make_pipeline(rfe, svr_rbf) # define Pipeline and call RFE and RideCV
    anova_svm.fit(features_train, labels_train)# fit the data in defined pipeline
    predict = np.zeros(labels_test.shape)
    predict=anova_svm.predict(features_test)# predict the labels of given features
    kaggle.kaggleize(predict, path+"Submission/Predictions/BankQueues/RidgeRegression_CrossValidated.csv") # save the file
    kaggle.kaggleize(predict, path+"Submission/Predictions/BankQueues/best.csv") #save the file as best.csv
    
    #Plot the graph of Regression Coefficient range[0.1+x,2+x] vs Errors
    plt.figure(3, figsize=(6,4))
    plt.plot(errors1.keys(),errors1.values(),'sb-', linewidth=3) #Plot the first series in blue with square marker
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Regression Coefficient range[0.1+x,2+x]") #X-axis label
    plt.title("Error vs Regression Coefficient for BankQueues") #Plot title
    #Save the chart
    plt.savefig(path+"/Submission/Figures/ErrorVsRegressionCoefficient_Bank.pdf")
    plt.show()
    
