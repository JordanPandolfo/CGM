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
from scipy import interpolate 

#Create normal density function
f = lambda r: norm.pdf(r,p.mu,p.sigr)

#Quadrature points and weights
ng = 4
xg, wg = leggauss(ng)

a = p.mu-3.5*p.sigr
b = p.mu+3.5*p.sigr
diff = b-a
absa = a +(xg+1)*diff/2
weight = diff/2*wg


"""
Value function estimation during retirement period
"""

import time
start_time = time.time()
#Retirement period (Index 45 to 80, Age 65 to 100): Fixed Income process
for i in range(1,35): #For each year of retirement

    #Create cubic interpolation of next-period value function
    x_grid_aug = np.append(np.log(x_grid),max(np.log(x_grid))+10)
    x_grid_aug = np.append(min(np.log(x_grid))-10,x_grid_aug) 
    V_aug = np.append(V[p.T-i,],V[p.T-i,p.xn-1])
    V_aug = np.append(V[p.T-i,0],V_aug)
        
    #Vp_interp = interp(x_grid_aug,V_aug,kind='cubic')
    Vp_interp = interp(x_grid_aug,V_aug,kind='linear')
    #tck = interpolate.splrep(x_grid_aug,V_aug)
    #Vp_interp = lambda x: interpolate.splev(x,tck)

    for ix, x in enumerate(x_grid):  #For each cash-in-hand value
        print(i,ix)
        
        #Setup on 3 dimensions for (cgrid,alphagrid,rgrid)
        hold = np.tile(x-c_grid,(len(alpha_grid),1)).T
        
        #if cash-on-hand less consumption negative, infeasible.  Set to zero.
        hold[hold < 0] = 0
        
        #Second step
        test = (np.ones((len(absa),len(c_grid),len(alpha_grid)))*hold).copy()
        
        #test1 = np.zeros(np.shape(test))
        #Final scaling
        for j in range(len(absa)):
            test[j] = test[j]*((1+absa[j])*alpha_grid+(1+p.rf)*(1-alpha_grid))+Retire_Income
            
        Interp = Vp_interp(np.log(test))
        
        Expected_value = np.zeros(( len(c_grid),len(alpha_grid) ))
        
        for j in range(len(absa)):
            Expected_value = (Expected_value + weight[j]*f(absa[j])*Interp[j]).copy()
        
        
        consume = c_grid.copy()
        consume[consume > x] = x.copy()
        
        value = (p.u(consume,alpha_grid)+p.delta*p.survprob[p.T-i]*Expected_value).copy()       
        
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

def income(t,nu,ep):
    """
    deterministic income, given age t
    t: age of agent
    nu, ep: persistent, transitory shocks
    """
    determine = p.b0+p.b1*45+p.b2*45**2+p.b3*45**3
    
    return np.exp(determine+nu+ep)*p.cpi

from scipy.stats import multivariate_normal

mean = np.array((p.mu,0,0))
cov = np.array(( 
        (p.sigr**2,0,0),
        (0,p.sigu**2,0),
        (0,0,p.sige**2)
        ))

F = lambda r,nu,ep : multivariate_normal.pdf(np.array((r,nu,ep)),mean,cov)

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
        
        #Setup on 3 dimensions for (cgrid,alphagrid,rgrid)
        hold = np.tile(x-c_grid,(len(alpha_grid),1)).T
        
        #if cash-on-hand less consumption negative, infeasible.  Set to zero.
        hold[hold < 0] = 0
        
        #Second step
        test = (np.ones((ng,nu,ne,len(c_grid),len(alpha_grid)))*hold).copy()
        #test = (np.ones((len(absa),len(c_grid),len(alpha_grid)))*hold).copy()

        #Final scaling to account for all R, nu, epsilon combinations 
        for j in range(ng):
            for k in range(nu):
                for l in range(ne):
                    
                    test[j,k,l] = (test[j,k,l]*( (1+absag[j])*alpha_grid+(1+p.rf)*(1-alpha_grid))                          
                          +income(65-i,absau[k],absae[l])  
                        )
                    
                    #test[j] = test[j]*((1+absa[j])*alpha_grid+(1+p.rf)*(1-alpha_grid))+Retire_Income
            
        Interp = Vp_interp(np.log(test))
        
        Expected_value = np.zeros(( len(c_grid),len(alpha_grid) ))
        
        for j in range(ng):
            for k in range(nu):
                for l in range(ne):
                    
                    Expected_value = (Expected_value + 
                                           weightg[j]*weightu[k]*weighte[l]*Interp[j,k,l]*F(absag[j],absau[k],absae[l])
                                               )
                    #Expected_value = (Expected_value + weight[j]*f(absa[j])*Interp[j]).copy()
        
        
        consume = c_grid.copy()
        consume[consume > x] = x.copy()
        
        value = (p.u(consume,alpha_grid)+p.delta*p.survprob[p.K-i]*Expected_value).copy()       
        
        value_max = (np.max(value)).copy()
        index = np.where(value == np.max(value))
        c_index, alpha_index = index[0][0], index[1][0]

        C[p.K-1-i,ix] = (c_grid[c_index]).copy()
        A[p.K-1-i,ix] = (alpha_grid[alpha_index]).copy()
        V[p.K-1-i,ix] = value_max.copy()
    

#import os
#os.chdir("/home/pando004/Desktop")
#np.savetxt("value2.csv",V,delimiter=",")
#np.savetxt("consume2.csv",C,delimiter=",")
#np.savetxt("alpha2.csv",A,delimiter=",")
  

