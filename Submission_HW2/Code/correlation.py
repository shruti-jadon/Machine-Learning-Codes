#correlation matrix
import numpy as np
#pass the location where data is stored
path = 'C:/Users/Shruti Jadon/Desktop/589_HW2/HW02/HW02/'
#Load the Data
data = np.load(path + 'Data/BankQueues/Data.npz')
features_train = data['X_train']
labels_train = data['y_train']
features_test = data['X_test']
labels_test = data['y_test']
print np.linalg.eig(np.corrcoef(features_train))

