import kaggle
import numpy as np
import matplotlib.pyplot as plt
from SupportVectorMachine import supportvector
from LogisticRegression import LR
from KNearestNeighbour import KNN
from sklearn.neighbors import KNeighborsClassifier
from Plot import plot
from KFoldCV import *

#import dataset
#kindly give path address of Submission Folder is Saved, in format "C:/Users/Shruti Jadon/Downloads"
path = 'C:/Users/Shruti Jadon/Desktop'
List=['Digits','Semiconductor','EmailSpam']
c=1
for i in List:
    data = np.load(path +'/Data/'+str(i)+'/Data.npz')
    features_train = data['X_train']
    labels_train = data['y_train']
    features_test = data['X_test']
    labels_test = data['y_test']
    
    #Array to store time taken for training and testing the dataset
    Training_time=[]
    Testing_time=[]
    
    SVM_Digits=supportvector(features_train,labels_train,features_test,labels_test)
    kaggle.kaggleize(SVM_Digits[0], path+"/Submission/Predictions/"+str(i)+"/SVM.csv")
    Training_time.append(SVM_Digits[1])
    Testing_time.append(SVM_Digits[2])
    print Training_time, Testing_time
    
    LR_Digits=LR(features_train,labels_train,features_test,labels_test)
    kaggle.kaggleize(LR_Digits[0], path+"/Submission/Predictions/"+str(i)+"/LR.csv")
    Training_time.append(LR_Digits[1])
    Testing_time.append(LR_Digits[2])
    print Training_time, Testing_time
    
    KNN_Digits=KNN(features_train,labels_train,features_test,labels_test)
    kaggle.kaggleize(KNN_Digits[0], path+"/Submission/Predictions/"+str(i)+"/KNN.csv")
    Training_time.append(KNN_Digits[1])
    Testing_time.append(KNN_Digits[2])
    print Training_time, Testing_time
    
    plot(c,Training_time,"Training Time",str(i),path)
    plot(c+3,Testing_time,"Testing Time",str(i),path)
    print "K Fold CrossValidation in Progress for "+ str(i)+" DataSet"
    
    # call Cross Validation function and saving its return error values in CV variable
    CV= kFold(features_train,labels_train,features_test,labels_test,path,str(i))
    
    #plot a graph of Neighbors vs Errors.
    values0 =CV[0].values() # save CrossValidation set Error Values
    inds0  =CV[0].keys()    # save CrossValidation set number of neighbor values
    values1 =CV[1].values() # save Training set Error Values
    inds1  =CV[1].keys()    # save respective number of neighbors for Training set Errors.
    plt.figure(c+10, figsize=(6,4))
    plt.plot(inds0,values0,'sb-', linewidth=3) #Plot the first series in blue with square marker
    plt.plot(inds1,values1,'or-', linewidth=3) #Plot the second series in red with square marker
    plt.ylabel("Error") #Y-axis label
    plt.xlabel("Neighbours") #X-axis label
    plt.title("Error vs Neighbours for "+str(i)+" Dataset") #Plot title
    labels =["CrossValidation Error","Training Error"]
    plt.legend(labels,loc="best")
    #Save the chart
    plt.savefig(path+"/Submission/Figures/CrossValidation_KNN"+str(i)+".pdf")
    plt.show()
    
    #Accuracy values on Kaggle, saved manually.
    value =[[10.689,89.074,97.447],[94.805,91.558,94.481],[81.989,93.370,80.442]]
    
    plot(c+20,value[c-1][:],"Accuracy on ",str(i),path)
    c=c+1