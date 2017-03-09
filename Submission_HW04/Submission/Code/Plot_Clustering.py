import matplotlib.pyplot as plt

def plotc(Scores, name,number,path):
    # Plot a bar chart of scores vs number of clusters
    plt.figure(number, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(Scores.keys(),Scores.values(),'or-', linewidth=1)  #line plot
    plt.grid(True) #Turn the grid on
    plt.ylabel("Silhouette Score") #Y-axis label
    plt.xlabel("Number of Clusters") #X-axis label
    plt.gca().set_xticks(Scores.keys()) #label locations
    plt.gca().set_xticklabels(Scores.keys()) #label values
    labels =[name+"'s Score"]  
    plt.legend(labels,loc="best")
    plt.tight_layout()
    plt.savefig(path+"/HW04/Figures/" + str(name) + ".pdf")  # save the figure
    plt.show()

def numberofclusters(path,n):
    inds=range(0,len(n))
    plt.figure(13, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds,n[:], align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Number of entries per Cluster") #Y-axis label
    plt.xlabel("Label of Cluster") #X-axis label
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(inds) #label values
    labels =["Number of Entries per Cluster"]  
    plt.legend(labels,loc="best")
    plt.tight_layout()
    plt.savefig(path+"/HW04/Figures/numberofclusters.pdf")  # save the figure
    plt.show()