#K-fold 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import kaggle
from sklearn.svm import SVR

 
def CrossValidation_Robot(features_train,labels_train,features_test,labels_test,path):
    n=features_train.shape
    k=2 # make it 2 fold 
    size = n[0]/k # size of each fold.
    errors1={}
    e1=[]
    errors2={}
    Cvalue=np.arange(100,500,100)
    degree=np.arange(1,n[1],2)
    for d in degree:
        for p in Cvalue:          # considering neighbours from range 1 to 20.
            print "Considering"+str(p)
            for i in range(1,k): 
                # Select the cross Validation set, it will change with changing values of i       
                Feature_CrossVal = features_train[i*size:][:size]  
                Label_CrossVal = labels_train[i*size:][:size]   
                
                # Add the rest of Training set  
                Feature_Train = features_train[:i*size]
                np.append(Feature_Train ,features_train[(i+1)*size:])  
                Label_Train = labels_train[:i*size]
                np.append(Label_Train ,labels_train[(i+1)*size:])
                #define Support Vector Regression with Degree and Regression Coefficient as Hyperparameter
                svr_rbf = SVR(kernel='rbf', C=p, degree=d)
                y_rbf = svr_rbf.fit(Feature_Train, Label_Train).predict(Feature_CrossVal)
                # Find the mean square error between the found predictions to the cross validation output set add error in a list 
                e1.append(np.sqrt(mean_squared_error(Label_CrossVal,y_rbf)))      
                print e1 
                # take mean of the K fold errors for particular neighbor and append it   
            errors1[p]=np.mean(e1) #saving absolute error with Value of regularization constant
        errors2[d]=np.mean(e1)     #saving absolute error with Value of Polynomial Degree
        
    #Find predictions with Min Error degree and Regression Coefficient
    print min(errors1, key=errors1.get)
    #define the Support Vector Regression  with optimum Regression Coefficient and Degree
    svr_rbf = SVR(kernel='rbf', C=min(errors1, key=errors1.get), degree=min(errors1, key=errors2.get))
    predict=np.zeros(labels_test.shape)
    #fit the data and predict the values
    predict = svr_rbf.fit(features_train, labels_train).predict(features_test)
    #save the output in CSV format
    kaggle.kaggleize(predict, path+"Submission/Predictions/RobotArm/SupportVectorRegression_CrossValidated.csv")
    kaggle.kaggleize(predict, path+"Submission/Predictions/RobotArm/best.csv")

    #Plot graph representing Regression coefficient vs Error
    plt.figure(1, figsize=(6,4))
    plt.plot(errors1.keys(),errors1.values(),'sb-', linewidth=3) #Plot the first series in blue with square marker
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Regression Coefficient") #X-axis label
    plt.title("Error vs Regression Coefficient for Robot Dataset") #Plot title
    #Save the chart
    plt.savefig(path+"/Submission/Figures/ErrorVsRegressionCoefficient_Robot.pdf")
    plt.show()
    
    #Plot graph representing Degree vs Error
    plt.figure(2, figsize=(6,4))
    plt.plot(errors2.keys(),errors1.values(),'or-', linewidth=3) #Plot the second series in red with square marker
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Degree") #X-axis label
    plt.title("Degree vs Mean Square Error for Robot Dataset") #Plot title
    #Save the chart
    plt.savefig(path+"/Submission/Figures/ErrorVsDegree_Robot.pdf")
    plt.show()
        

