#Metropolis of all types
import numpy as np
import Basic as b
import plotting as p
import random


#(f.)
def MetropolisJsampling(J,B,alpha):    # Metropolis sampling algorithm for Jars
    Jmean=np.array([0]*len(J))         # declare a J mean.
    #Jtemp=np.array([0]*len(J))
    for i in range(1,1000):           # run the iteration for 1000 times.
        Jnew=b.Jrandom(J)               # Randomize the Jar sequence Value
        acceptance_ratio=b.PalphaJB(alpha,Jnew,B)/b.PalphaJB(alpha,J,B) # find Acceptance Ratio, which is division of probabilities with new vs old values 
        if random.uniform(0,1)<= acceptance_ratio:  # choose a random number, if It is less than acceptance ratio
            J[:]=Jnew[:]                        # if it is less than, then assign it to a temp varaibale 
        Jmean[:]=np.add(J[:],Jmean[:])          # add temp variable to Jmean entries
    return [x/1000.0 for x in Jmean[:]]            # return mean of all entries for J values.


def MetropolisAlphaSampling(J,B,alpha):   # Metropolis Sampling Algorithm for alpha
    alpha_mean=0.0                         # declare a variable which will add all values of alpha, after sampling
    alpha_collect=np.array([]) 
    for i in range(1,3000):                # run iterations till 3000
        a1=b.PalphaJB(alpha,J,B)
        alpha_new=b.Randomalpha(alpha)       # find alpha randomly
        b1=b.PalphaJB(alpha_new,J,B)
        acceptance_ratio=b1/a1               # find Acceptance Ratio, which is division of probabilities with new vs old values 
        if np.random.rand()<= acceptance_ratio: # if random number found is less than accepatance ratio then assign alpha to new alpha
            alpha=alpha_new
        alpha_collect = np.append( alpha_collect , alpha ) 
        alpha_mean+=alpha                    # add alpha values to alpha mean
    return alpha_mean/3000,alpha_collect                   # return mean values of alpha_mean 


#(i.)
def MetropolisAlphaJ(J,B,alpha):        #Metropolis Algorithm for J and Alpha sampling
    alpha_mean=0.0                      # set a variable to 0, in which all alpha values will get accumulated
    alpha_collect=np.array([])
    Jmean=np.array([0]*len(J))          # set a variable to 0, in which all J values will get accumulated
    for i in range(1,5000):             # Iterate 5000 times 
        random_values=b.genalphaJ(alpha,J)    # generate random values of both alpha and J
        acceptance_ratio=b.PalphaJB(random_values[0],random_values[1],B)/b.PalphaJB(alpha,J,B)    # generate acceptance ratio for alpha
        if random.uniform(0,1)<= acceptance_ratio:    # if randomly generated value is less than acceptance ratio, then assign random alpha to alpha.
            alpha=random_values[0]
            J[:]=random_values[1]
        alpha_collect = np.append( alpha_collect , alpha )                
        alpha_mean+=alpha 
        # accumulate values of alpha to alpha_mean
        Jmean[:]=np.add(J[:],Jmean[:])  # accumulate values of J, to Jmean numpy array.
    return alpha_mean/5000, [x/5000.0 for x in Jmean[:]], alpha_collect # return both sampled alpha mean and Jmean


