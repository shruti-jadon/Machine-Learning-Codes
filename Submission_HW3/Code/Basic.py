import numpy as np
import random
import plotting as p

#(a.) Probability of Jar given alpha. where alpha is probability of chosing a same Jar.
J=np.array([])
def PJgivenalpha(J,alpha):
    PJgivenalpha=1.0  # define a variable, and set to 1.0
    if (J[0]==1):     # check condition, if first chosen Jar is 1, then return 0.
        return 0      
    else:             # if Jar 0 is chosen at first
        for i in range(0,len(J)-1): # run a loop till second last chosen Jar
            if(J[i]==J[i+1]):       # if current Jar is equal to next chosen Jar      
                PJgivenalpha*=alpha # alpha will be multiplied, as both
            else:
                PJgivenalpha*=(1-alpha)
        return PJgivenalpha   # return product of all independent probabilities
        

#(b.)
def PBgivenJ(J,B):
    PBgivenJ=1.0
    ProbJB=np.array([[0.2,0.8],[0.9,0.1]]) #[0][0] indicates Jar 0 with white ball, [0][1] means Jar 0 black ball
    for i in range(0,len(J)):             # Run Loop till length of any of B and J
        PBgivenJ*=ProbJB[J[i]][B[i]]       # Finding probability of a specific jar and specific ball from that jar. 
    return PBgivenJ                       # return probabilities of all independent     

#(c.)
def Palpha(alpha):
    if(alpha>=0 and alpha<=1):    # put condtition of alpha is greater than 0 and less than 1
        return 1                  # approve it, if its satisfies condition
    else:
        return 0                  # disapproves if it doesn't meet the condition

#(d.)
def PalphaJB(alpha,J,B):               # function to find probability of alpha given Jar and Ball
    probability=Palpha(alpha)*PBgivenJ(J,B)*PJgivenalpha(J,alpha)     # considering Bayes rule, it will be multiplication of all above functions 
    return probability              # return multiplication of all probability function by bayes rule.

#(e.)
def Jrandom(J):         # function to find random Jar selection
    Jout=np.array(J) 
    i=random.randint(0,len(J)-1)  # randomly chose the position of Jth given array  
    if(Jout[i]==0):               # if random position chosen was jar 0 make it jar 1 and vice versa
        Jout[i]=1
    else:
        Jout[i]=0
    return Jout                 # Flip the random position's element and return it.


#(g.)
def Randomalpha(alpha):             # declare function for Random alpha
    return np.random.rand()         # return Random number between 0 and 1

#(h.)
def genalphaJ(alpha,J):                    # generate both random alpha, J sequence
    return Randomalpha(alpha), Jrandom(J)  # call function of alpha and J randomization


def ZeroOneMode(output):                # function to convert values to 1 if it is greater than 0.5 and 0 if it is less than 0.5
    ZeroOneMode=np.array(output[1])
    for i in range(0,ZeroOneMode.size):
        if(ZeroOneMode[i]<0.5):
            ZeroOneMode[i]=0
        else:
            ZeroOneMode[i]=1
    return ZeroOneMode
            
