import numpy as np
import matplotlib.pyplot as plt
import util
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import AgglomerativeClustering
import Plot_Clustering

path="C:/Users/Shruti Jadon/Documents"
#Load the data. imgs is an array of shape (N,8,8) where each 8x8 array
#corresponds to an image. imgs_vectors has shape (N,64) where each row
#corresponds to a single-long-vector respresentation of the corresponding image.
img_size = (8,8)
imgs, imgs_vectors = util.loadDataQ1()
print imgs_vectors

Scores={}  # define a dictionary to store values and scores
for v in range(2,15):
    model = AgglomerativeClustering(n_clusters=v) # define clustering model 
    model.fit(imgs_vectors)                       # fit the model
    Zs=model.labels_
    Scores[v]= round(silhouette_score(imgs_vectors, Zs, metric='euclidean'),2) # store the scores of model

Plot_Clustering.plotc(Scores,"Agglomerative Clustering",11,path)

# determine actual lables and number of points assigned to them using optimal number of clusters
model = AgglomerativeClustering(n_clusters=9) # declare model
model.fit(imgs_vectors) #fit the model
Zs=model.labels_ # store labels

# store number of points assigned to each cluster.
n=[0]*11
for i in range(0,len(Zs)):
    n[Zs[i]]+=1

# Bar Plotting of number of points assigned to each cluster vs number of clusters    
Plot_Clustering.numberofclusters(path,n)

# Experimentation with K means
for v in range(2,15):
    kmeans = KMeans(n_clusters=v, random_state=0).fit(imgs_vectors)
    print kmeans.labels_   
    Zs=kmeans.labels_
    Scores[v]= round(silhouette_score(imgs_vectors, Zs, metric='euclidean'),2)

# plotting of experimenation clustering error
Plot_Clustering.plotc(Scores,"K means Clustering",12,path)

# determine actual lables and number of points assigned to them using optimal number of clusters
model = AgglomerativeClustering(n_clusters=9) # declare model
model.fit(imgs_vectors) #fit the model
Zs=model.labels_ # store labels

# plotting of digits
K=10
N  = imgs_vectors.shape[0]
#The code below shows how to plot examples from clusters as an image array
for k in np.unique(Zs):
	plt.figure(k)
	if np.sum(Zs==k)>0:
  	  util.plot_img_array(imgs_vectors[Zs==k,:], img_size,grey=True)
  	plt.suptitle("Cluster Exmplars %d/%d"%(k,K))
plt.show()

#The code below shows how to compute and plot cluster centers as an image array
centers = np.zeros((len(np.unique(Zs)),64))
plt.figure(1)

i=0
for k in np.unique(Zs):
    centers[i,:] = np.mean(imgs_vectors[Zs==k,:],axis=0)
    i=i+1
util.plot_img_array(centers, img_size,grey=True)
plt.suptitle("Cluster Centers (K=%d)"%K)
plt.show()