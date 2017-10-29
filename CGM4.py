"""
Author: Jordan Pandolfo

Replicating Cocco, Gomes and Maenhout 'Consumption and Portfolio Choice 
over the Life Cycle,' REview of Financial Studies 2005.

Work with calibration for high school graduates
"""

import CGMparam
p = CGMparam   #Rename module to save space in code

import pylab
import numpy as np
import matplotlib.pyplot as plt

"""
Backward induction on the value function and policy function
Decision variables: consumption and portfolio choice discretized over
    an equally-spaced grid
    
State space: cash-on hand (X_t)

Upper and lower bounds for cash-on-hand and consumption chosen to be 
    non-binding
    
Gaussian Quadrature: used by authors on the risky asset return, as well as
    for the two normal innovations to the labor income process
    
Cubic Spline Interpolation: Used on the log of the state variable X_t
    to pick up precautionary savings motives.
"""    
#Set grid for share of wealth invested in risky, risk-free assets
alpha_grid = np.linspace(p.alphaa,p.alphab,p.alphan)
#Set grid for consumption policy function
c_grid = np.linspace(p.ca+.001,p.cb,p.cn)
#Set grid for state variable, cash-in hand
x_grid = np.linspace(p.xa+.001,p.xb,p.xn)

#Value function over time T and state space X
V = np.zeros((p.T,p.xn))
#Consumption policy functions
C = np.zeros((p.T,p.xn))
#Alpha policy function
A = np.zeros((p.T,p.xn))

#Terminal period (Index 80, Age 100): Agent consumes everything
Retire_Income = np.log(p.lamda)+p.regress(p.K+20)+0.0 #fixed retirement payout, with zero shock
#Retire_Income = np.exp(p.b0+p.b1*45+p.b2*45**2+p.b3*45**3+np.log(p.lamda))

#V[p.T-1,] = (x_grid+Retire_Income)**(1-10)/(1-10)
V[p.T-1,] = (x_grid)**(1-p.gamma)/(1-p.gamma)
#connsume everything, invest nothing
#C[p.T-1,] = x_grid #+Retire_Income
#A[p.T-1,] = np.zeros(p.xn)

#V[p.T-1,0] = 0

from scipy.interpolate import interp1d as interp
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss    

#Create normal density function
f = lambda r: norm.pdf(r,p.mu,p.sigr)

#Quadrature points and weights
ng = 4
xg, wg = leggauss(ng)

a = p.mu-3.5*p.sigr
b = p.mu+3.5*p.sigr
diff = b-a
absa = a +(xg+1)*diff/2
#absam = matrix(absa)
weight = diff/2*wg


"""
Value function estimation during retirement period
"""


probweight = weight*f(absa)
probweightp = np.tensordot(probweight,np.ones((p.cn,p.alphan)) , axes = 0)


import time
start_time = time.time()
#Retirement period (Index 45 to 80, Age 65 to 100): Fixed Income process
for i in range(1,35): #For each year of retirement

    #Create cubic interpolation of next-period value function
    x_grid_aug = np.append(np.log(x_grid),max(np.log(x_grid))+10)
    x_grid_aug = np.append(min(np.log(x_grid))-10,x_grid_aug) 
    V_aug = np.append(V[p.T-i,],V[p.T-i,p.xn-1])
    V_aug = np.append(V[p.T-i,0],V_aug)
        
    #Vp_interp = interp(x_grid_aug,V_aug,kind='linear')
    Vp_interp = interp(x_grid_aug,V_aug,kind='linear')
    
    for ix, x in enumerate(x_grid):  #For each cash-in-hand value
        print(i,ix)
        
        #Setup on 3 dimensions for (cgrid,alphagrid,rgrid)
        hold = np.tile(x-c_grid,(len(alpha_grid),1)).T
        
        #if cash-on-hand less consumption negative, infeasible.  Set to zero.
        hold[hold < 0] = 0
        
        #Second step
        test = (np.ones((len(absa),len(c_grid),len(alpha_grid)))*hold).copy()
                
        alpha_space = np.tile(alpha_grid,(p.cn,1))
        alpha_spacep = np.tensordot(absa,alpha_space,axes=0) + alpha_grid+(1+p.rf)*(1-alpha_grid)
        testp = test*alpha_spacep + Retire_Income
    
        #Final scaling
        #for j in range(len(absa)):
        #    test[j] = test[j]*((1+absa[j])*alpha_grid+(1+p.rf)*(1-alpha_grid))+Retire_Income
            
        Interp = Vp_interp(np.log(testp))
        
        #Expected_value = np.zeros(( len(c_grid),len(alpha_grid) ))
        Expected_value_el = probweightp*Interp
        Expected_valuep = np.sum(Expected_value_el,axis = 0)
        
        #for j in range(len(absa)):
        #    Expected_value = (Expected_value + weight[j]*f(absa[j])*Interp[j]).copy()
        
        
        consume = c_grid.copy()
        consume[consume > x] = x.copy()
        
        value = (p.u(consume,alpha_grid)+p.delta*p.survprob[p.T-i]*Expected_valuep).copy()       
        
        value_max = (np.max(value)).copy()
        index = np.where(value == np.max(value))
        c_index, alpha_index = index[0][0], index[1][0]

        C[p.T-1-i,ix] = (c_grid[c_index]).copy()
        A[p.T-1-i,ix] = (alpha_grid[alpha_index]).copy()
        V[p.T-1-i,ix] = value_max.copy()
    
print("--- %s seconds ---" % (time.time() - start_time))

"""
Value function estimation during working period
"""

from scipy.stats import multivariate_normal
import itertools

mean = np.array((p.mu,0,0))
cov = np.array(( 
        (p.sigr**2,0,0),
        (0,p.sigu**2,0),
        (0,0,p.sige**2)
        ))

#Ordering of inputs is r, nu, epsilon
F = lambda x : multivariate_normal.pdf(np.array((x[0],x[1],x[2])),mean,cov)

#3d quadrature weights and nodes
ng, nu, ne = 4,4,4
xg, wg = leggauss(ng)
xu, wu = leggauss(nu)
xe, we = leggauss(ne)

ag = p.mu-3.5*p.sigr
bg = p.mu+3.5*p.sigr
diffg = bg-ag
absag = ag +(xg+1)*diffg/2
weightg = diffg/2*wg

au = -3.5*p.sigu
bu = 3.5*p.sigu
diffu = bu-au
absau = au +(xu+1)*diffu/2
weightu = diffu/2*wu

ae = -3.5*p.sige
be = 3.5*p.sige
diffe = be-ae
absae = ae +(xe+1)*diffe/2
weighte = diffe/2*we

#total = 0
#for i in range(ng):
#    for j in range(nu):
#        for k in range(ne):
#            total = total + weightg[i]*weightu[j]*weighte[k]*F(absag[i],absau[j],absae[k])


#Create list of weight terms involved
columnsp = [
        list(weightg),
        list(weightu),
        list(weighte)
        ]
# ng*nu*ne x 3 list of weights.
#   All combinations 
holderp = list(itertools.product(*columnsp))
#Multiply each combination for 'joint weight'
weight_vector = np.prod(holderp,axis=1)
        
#Compute abscissa at each grid point
columnspp = [
        list(absag),
        list(absau),
        list(absae)
        ]
# ng*nu*ne x 3 list of abscissa
#   Same ordering as weights
holderpp = list(itertools.product(*columnspp))

Fload = np.zeros((np.shape(holderpp)[0],1))

#Compute probability of combination
for i in range(np.shape(holderpp)[0]):
    Fload[i] = F(holderpp[i])
    
#Compute product of joint probbaility and joint weight for each combination    
Full_weight = weight_vector*Fload[:,0] 
#Apply each probability/weight to a cn x an sized array.  
Full_weight = np.tensordot(Full_weight,np.ones((p.cn,p.alphan)),axes=0)

#Working period (Index 0 to 45, Age 20-65): Stochastic Income Process 
#Retirement period (Index reads 45 to 80 in cell, corresponds to age 65
#   to 100).  During this time, there is a fixed income process
for i in range(45):
    
    #Create cubic interpolation of next-period value function
    x_grid_aug = np.append(np.log(x_grid),max(np.log(x_grid))+10)
    x_grid_aug = np.append(min(np.log(x_grid))-10,x_grid_aug) 
    V_aug = np.append(V[p.K-i,],V[p.K-i,p.xn-1])  #K: retirement age
    V_aug = np.append(V[p.K-i,0],V_aug)
    
    #Vp_interp = interp(x_grid_aug,V_aug,kind='cubic')
    Vp_interp = interp(x_grid_aug,V_aug,kind='linear')
    
    for ix, x in enumerate(x_grid):
        print(i,ix)
        
        #Given x, compute x-c for every c choice
        #   Repeat for alphan columns
        hold = np.tile(x-c_grid,(len(alpha_grid),1)).T
        
        #if cash-on-hand less consumption negative, infeasible.  Set to zero.
        hold[hold < 0] = 0
        
        #Second step
        #test = (np.ones((ng,nu,ne,len(c_grid),len(alpha_grid)))*hold).copy()
        #Apply hold to each sheet, representing each 'r' return outcome
        test = (np.ones((ng,p.cn,p.alphan))*hold).copy()
        
        #Every alpha decision repeated for cn rows.
        alpha_space = np.tile(alpha_grid,(p.cn,1))
        
        #Each page represents a return outcome 'r'
        #   Within page, the alpha decisions repeated over cn rows
        #   Matches up with dimension of 'test'
        alpha_spacep = np.tensordot(absag,alpha_space,axes=0) + alpha_grid+(1+p.rf)*(1-alpha_grid)
        
        #Element by element multiplication
        #   (x-c)*R where each page represents different r outcome
        testp = test*alpha_spacep
        
        #Creates nu*ne duplicates
        #   First nu*ne pages are testp[0]
        #   Second nu*ne pages are testp[1]
        testp_clone = np.repeat(testp,nu*ne,axis=0)
        
        #Deterministic age income component
        determine = p.b0+p.b1*(65-i)+p.b2*(65-i)**2+p.b3*(65-i)**3
        
        columns = [
                list(absau),
                list(absae)
                ]
                
        #Joint value of income shock
        #   Summed, not multiplied
        #   Same ordering of combinations between nu and epsilon
        holder = np.sum(list(itertools.product(*columns)),axis=1)        
        
        income_value = p.cpi*np.exp(determine+holder)
        
        #income_valuep = income_value.repeat(ng)
        #stacks incoem value ng times
        income_valuep = np.tile(income_value,ng)
        
        testpp = testp_clone + np.tensordot(income_valuep,np.ones((p.cn,p.alphan)),axes=0)
        
        Interpp = Vp_interp(np.log(testpp))
        
        Expected_value_pre = Full_weight*Interpp
        Expected_valuep = np.sum(Expected_value_pre,axis = 0)
        
        consume = c_grid.copy()
        consume[consume > x] = x.copy()
        
        value = (p.u(consume,alpha_grid)+p.delta*p.survprob[p.K-i]*Expected_valuep).copy()       
        
        value_max = (np.max(value)).copy()
        index = np.where(value == np.max(value))
        c_index, alpha_index = index[0][0], index[1][0]

        C[p.K-1-i,ix] = (c_grid[c_index]).copy()
        A[p.K-1-i,ix] = (alpha_grid[alpha_index]).copy()
        V[p.K-1-i,ix] = value_max.copy()
        
#import os
#os.chdir("/home/pando004/Desktop")
#np.savetxt("value.csv",V,delimiter=",")
#np.savetxt("consume.csv",C,delimiter=",")
#np.savetxt("alpha.csv",A,delimiter=",")

