# -*- coding: utf-8 -*-
import numpy as np
import Basic as b
import plotting as p
import random

def random_alpha_Experiment(alpha):
    #i=np.random.logistic(loc=alpha, scale=1.0, size=None)
    i=np.random.normal(loc=0.5, scale=5, size=None)
    if (i>=1 or i<=0):
       return random_alpha_Experiment(alpha)
    else:
        return i
J=np.array([0,0,0,0,0])
B=np.array([1,1,1,1,1])

def MCAlphaExperiment(J,B,alpha):        #Metropolis Algorithm for J and Alpha sampling
    alpha_mean=0.0                      # set a variable to 0, in which all alpha values will get accumulated
    alpha_collect=np.array([])        # set a variable to 0, in which all J values will get accumulated
    for i in range(1,3000):             # Iterate 5000 times 
        random_values=b.genalphaJ(alpha,J)    # generate random values of both alpha and J
        acceptance_ratio=b.PalphaJB(random_alpha_Experiment(alpha),J,B)/b.PalphaJB(alpha,J,B)    # generate acceptance ratio for alpha
        if random.uniform(0,1)<= acceptance_ratio:    # if randomly generated value is less than acceptance ratio, then assign random alpha to alpha.
            alpha=random_values[0]            
        alpha_collect = np.append( alpha_collect , alpha )                
        alpha_mean+=alpha 
        # accumulate values of alpha to alpha_mean
    return alpha_mean/3000, alpha_collect # return both sampled alpha mean and Jmean
    
def random_Jar_Experiment(J):
    Jout=np.array(J)
    l=random.randint(0,(len(J)-1)/2) # randomly chose how many bits should be flipped
    for k in range(0,l):
        i=random.randint(0,len(J)-1)  # randomly chose the position of Jth given array  
        if(Jout[i]==0):               # if random position chosen was jar 0 make it jar 1 and vice versa
            Jout[i]=1
        else:
            Jout[i]=0
    return Jout  
    
def MC_Jar_Experiment(J,B,alpha):
    Jmean=np.array([0]*len(J))         # declare a J mean.
    #Jtemp=np.array([0]*len(J))
    for i in range(1,500):           # run the iteration for 1000 times.
        Jnew=random_Jar_Experiment(J)               # Randomize the Jar sequence Value
        acceptance_ratio=b.PalphaJB(alpha,Jnew,B)/b.PalphaJB(alpha,J,B) # find Acceptance Ratio, which is division of probabilities with new vs old values 
        if random.uniform(0,1)<= acceptance_ratio:  # choose a random number, if It is less than acceptance ratio
            J[:]=Jnew[:]                        # if it is less than, then assign it to a temp varaibale 
        Jmean[:]=np.add(J[:],Jmean[:])          # add temp variable to Jmean entries
    return [x/500.0 for x in Jmean[:]]   

   
      

 
    
    
