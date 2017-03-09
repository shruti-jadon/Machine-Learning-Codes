import numpy as np
import matplotlib.pyplot as plt
import Basic as b

def Histogram(values,path,types):
    plt.figure(1, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.hist(values,bins="auto", normed=False)
    plt.xlabel("Sampled Alpha Value") #X-axis label
    plt.ylabel("Alpha Distribution")
    plt.tight_layout()
    plt.title("sampled alpha values over iterations for "+str(types)) #Plot title
    plt.savefig(str(path)+"Submission/figures/"+str(types)+"_alpha.pdf")
    plt.show()

def BarGraph(Values,path,types): 
    #Create values and labels for line graphs
    inds = np.arange(len(Values))
    plt.figure(2, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.bar(inds, Values, align='center') #Plot the first series in red with circle marker
    #This plots the data
    plt.grid(True) #Turn the grid on
    plt.ylabel("Sampled Jar Values") #Y-axis label
    plt.xlabel("Sequence of Jar") #X-axis label
    plt.tight_layout()
    plt.title("Sampled Jar Values for "+str(types)) #Plot title
    plt.xlim(0,len(Values)) #set x axis range
    plt.ylim(0,2) #Set yaxis range    
    #Make sure labels and titles are inside plot area
    plt.tight_layout()   
    #Save the chart
    plt.savefig(str(path)+"Submission/figures/"+str(types)+"_Jar.pdf")  
    #Displays the plots.
    plt.show()

