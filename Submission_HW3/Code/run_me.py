import numpy as np
import Basic as b
import plotting as p
import experiments as e
import MCMC as m
import matplotlib.pyplot as plt

# pass the path where Submission folder is saved.
path="C:/Users/Shruti Jadon/Desktop/590_HW3/"
J=np.array([0,1,1,0,0,0,1,0]) 
B=np.array([1,1,0,1,1,0,0,0]) 


#(F.) output
J=np.array([0,0,1,1,1])    # pass the J, B and alpha Values
B =np.array ([1,1,0,0,1])
alpha=0.6
output=m.MetropolisJsampling(J,B,alpha)  # call metropolis J sampling function of normal distribution, and normal J sampling by chosing one random bit flipping
print "output of (f.) question. i.e. Jar Values"
print output            # print values in console
p.BarGraph(output,path,"output of (f.) question")    # plot graph of 

#(F.) running Experiment of J_random to check if we get same output or not.
output_experiment=e.MC_Jar_Experiment(J,B,alpha)  # calling experiment metropolis Sampling question
print "output_experiment values"
print output_experiment
p.BarGraph(output_experiment,path,"Experimental output of (f.) question") # plotting graph of Experiment

#(H.) output
J=np.array([0,1,1,0,1]) 
B =np.array([1,0,0,1,1]) 
alpha=0.7
output_alpha=m.MetropolisAlphaSampling(J,B,alpha)  # call function of alpha sampling
print"alpha mean value is"+str(output_alpha[0])     
p.Histogram(output_alpha[1],path,"output of (h.) question")   # plot histogram of alpha sampling

#(H.) running Experiment of alpha_random to check if we get same output
output_e_alpha=e.MCAlphaExperiment(J,B,alpha)
print "alpha mean value through experiment is"+str(output_e_alpha[0])
p.Histogram(output_alpha[1],path,"output experiment of (h.) question")


#(J.) output for uniform distribution question
J=np.array([0,1,1,0,0,0,1,0]) 
B=np.array([1,1,0,1,1,0,0,0]) 
alpha=0.5
output_Jalpha=m.MetropolisAlphaJ(J,B,alpha)
print "Jar mean value is" +str(output_Jalpha[1])
p.BarGraph(output_Jalpha[1][:],path,"output of (J.) question")    # plot Bar Graph of Jar sampling
print "alpha mean value is"+str(output_Jalpha[0])
p.Histogram(output_Jalpha[2],path,"output of (J.) question")      # plot histogram of alpha sampling


















