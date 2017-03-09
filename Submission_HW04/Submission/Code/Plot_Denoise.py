import matplotlib.pyplot as plt

def plot(errors,name,number,path):
    plt.figure(number, figsize=(6,4))  #6x4 is the aspect ratio for the plot
    plt.plot(errors.keys(),errors.values(),'or-', linewidth=1)  # line plot of errors vs different hyperparameter values
    plt.grid(True) #Turn the grid on
    plt.ylabel("Error of Denoised to Clean for "+str(name)) #Y-axis label
    plt.xlabel("Number of Components") #X-axis label
    plt.tight_layout()
    plt.savefig(path+"/HW04/Figures/" + str(name) + ".pdf") # save figure
    plt.show()    # show plot
    