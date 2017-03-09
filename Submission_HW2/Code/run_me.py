import kaggle
import numpy as np
from CrossValidation_Robot import CrossValidation_Robot
from CrossValidation_Bank import CrossValidation_Bank
from OtherMethods import Experiment_DataSet
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

#pass the path where Data is stored.
path = 'C:/Users/Shruti Jadon/Desktop/'
#Load the Data
data = np.load(path + 'Data/RobotArm/Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
feature_train=scale(features_train.data)

# calling the CrossValidation Function of SupportVector Regression on RobotArm DataSet
CrossValidation_Robot(features_train,labels_train,features_test,labels_test,path)

#calling other trial methods and saving file in predictions
Experiment_DataSet(features_train,labels_train,features_test,labels_test,path,"RobotArm")

#Load the BankQueues Data
data = np.load(path + 'Data/BankQueues/Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']

# calling the CrossValidation Function of Ridge Regression on BankQueues DataSet
CrossValidation_Bank(features_train,labels_train,features_test,labels_test,path)

#calling other trial methods and saving file in predictions
Experiment_DataSet(features_train,labels_train,features_test,labels_test,path,"BankQueues")

#Values obtained using Kaggle Submissions
values   = [1.03756,1.28544,1.04149]
inds   =np.arange(3)
labels = ["SVR with RBF","Ridge with CrossValidation","SVR with selectK"] # define set of classifiers as sequence as they have been called in main function
# Plot a bar chart
plt.figure(4, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.bar(inds, values, align='center') #This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error Rate") #Y-axis label
plt.xlabel("Method") #X-axis label
plt.gca().set_xticks(inds) #label locations
plt.gca().set_xticklabels(labels) #label values
plt.tight_layout()
plt.title("Error Rate vs Method for RobotArm Dataset") #Plot title
#Save the chart 
plt.savefig(str(path)+"/Submission/Figures/MethodError_RobotArm.pdf")
plt.show()

values   = [0.05183, 0.04780,0.04127, 0.04286]
inds   =np.arange(4)
labels = ["SVR Linear","SVR_RBF","LassoCV with RFE","LassoCV with SelectK"] # define set of classifiers as sequence as they have been called in main function
# Plot a bar chart
plt.figure(5, figsize=(6,4))  #6x4 is the aspect ratio for the plot
plt.bar(inds, values, align='center') #This plots the data
plt.grid(True) #Turn the grid on
plt.ylabel("Error Rate") #Y-axis label
plt.xlabel("Method") #X-axis label
plt.gca().set_xticks(inds) #label locations
plt.gca().set_xticklabels(labels) #label values
plt.tight_layout()
plt.title("Error Rate vs Method for BankQueues Dataset") #Plot title
#Save the chart 
plt.savefig(str(path)+"/Submission/Figures/MethodError_BankQueues.pdf")
plt.show()

