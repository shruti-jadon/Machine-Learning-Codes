import matplotlib.pyplot as plt
import numpy as np

def plot(num,Values,Title,name,address):
    inds   =np.arange(3)
    labels = ["SVM","LR","KNN"] # define set of classifiers as sequence as they have been called in main function
    # Plot a bar chart
    plt.figure(num, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, Values, align='center') #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel(str(Title)) #Y-axis label
    plt.xlabel("Method") #X-axis label
    plt.gca().set_xticks(inds) #label locations
    plt.gca().set_xticklabels(labels) #label values
    plt.tight_layout()
    plt.title(str(Title)+" vs Method for "+str(name)+"Dataset") #Plot title
    #Save the chart  
    Name=str(Title)+"_"+str(name)
    plt.savefig(str(address)+"/Submission/Figures/" +Name+".pdf")
    plt.show()