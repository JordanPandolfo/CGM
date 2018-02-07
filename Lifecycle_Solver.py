"""
@author: pando004
"""

#-------------------------------#
#                               #
#   Import Packages and Data    #
#                               #
#-------------------------------#
from scipy.optimize import minimize
import Lifecycle_data
p = Lifecycle_data   

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
from scipy.interpolate import interp1d as interp
from scipy.stats import norm
from numpy.polynomial.legendre import leggauss 
import os   
import itertools
from numpy import genfromtxt
from scipy.optimize import differential_evolution

#---------------------------------------------#
#                                             #
#   Parameter Values and Grid Construction    #
#                                             #
#---------------------------------------------#
#Distribution of retirement period stochastic processes
f = lambda r: norm.pdf(r,mu,sigr)

#Distribution of working period stochastic processess
F = lambda r,eps: multivariate_normal.pdf(np.array((r,eps)),mean,cov)


delta    = 0.96
gamma    = 10
rf       = .02
mu       = .06    
eps_mu   = 0
sigr     = 0.157 
sige     = .07
rho      = 0
K        = 45    
T        = 80     

scale = 2.3
b0 = -2.170042+2.700381 #labor income regression intercept
b1 = 0.16818  #Age
b2 = -0.0323371/10 #Age**2/10
b3 = 0.0019704/100 #Age**3/100

Retire_Income = 25 #fraction*working(65,eps_mu)  

#Grid points   
xa, xb, xn         = 7.5, 2000, 2000
x_grid     = np.linspace(xa,xb,xn)

#Quadrature points and weights
ng, ne  = 60,60
xg, wg  = leggauss(ng)
xe, we  = leggauss(ne)

ag      = mu-3.5*sigr
bg      = mu+3.5*sigr
diffg   = bg-ag
absag   = ag +(xg+1)*diffg/2
weightg = diffg/2*wg

ae      = eps_mu-3.5*sige
be      = eps_mu+3.5*sige
diffe   = be-ae
absae   = ae +(xe+1)*diffe/2
weighte = diffe/2*we

mean    = np.array((mu,eps_mu))
cov     = np.array(( 
                     (sigr**2,rho*sige*sigr),
                     (rho*sige*sigr,sige**2)
                   ))

#For numerical quadrature in retirement
quad_vector = weightg*f(absag)
return_vector = 1+absag

#For numerical quadrature in working period
columnsp = [list(absae),list(absag)]
holderp = list(itertools.product(*columnsp))    #Ordered combinations of labor and market shocks

eps_vector = np.array(holderp)[:,0]

return_vector_working  = 1+np.array(holderp)[:,1]

columnspp = [list(weighte),list(weightg)]
holderpp = list(itertools.product(*columnspp))
joint_weight = np.array(holderpp)[:,0]*np.array(holderpp)[:,1]

joint_prob = np.zeros(len(joint_weight))

for i in range(len(joint_weight)):
    joint_prob[i] = F(holderp[i][1],holderp[i][0])

joint = joint_weight*joint_prob

#-----------------------#
#                       # 
#   Custom Functions    #
#                       # 
#-----------------------#

def working(age,epsilon):
    """
    Inputs: 1) Age of worker
            2) Gross percentage shock to income
    
    Return: Labor income
    """
    
    return np.exp(b0+b1*age+b2*age**2+b3*age**3+epsilon)*scale  #*2 #1.5  #*scale  
    #return 25.0

def u(c):
    return c**(1-gamma)/(1-gamma)


def Vp_Retired(x,a,c,pi,Interp):
    """
    Inputs: 1) Cash-in-hand
            2) Portfolio share of risky asset
            3) Consumption
            4) Mortality probability
            5) Interpolating function
    Return: utility + discounted value function
    """
    utility = u(c)
    
    discount = delta*pi
    
    expected_value = np.sum(quad_vector*Interp( np.log( Retire_Income + (x-c)*(a*return_vector+(1-a)*(1+rf)  ) ) ) )
    
    return -(utility + discount*expected_value)    

def Vp_Working(x,a,c,pi,Interp,age):
    """
    Inputs: 1) Cash-in-hand
            2) Portfolio share of risky asset
            3) Consumption
            4) Mortality probability
            5) Interpolating function
            6) Age of worker 
    Return: utility + discounted value function
    """
    utility = u(c)
    
    discount = delta*pi
    
    expected_value = np.sum(  
            joint*Interp( np.log(
                        working(age,eps_vector) + (x-c)*(a*return_vector_working+(1-a)*(1+rf)  ) 
                                 ) ) 
            )
    
    return -(utility + discount*expected_value)


#-----------------#
#                 #
#   Initialize    #
#                 #
#-----------------#
V = np.zeros((T,xn))
C = np.zeros((T,xn))
A = np.zeros((T,xn))

V[T-1] = (x_grid)**(1-gamma)/(1-gamma) 

#---------------------#
#                     #
#   Estimate Model    #
#                     #
#---------------------#

for age in range(T-1,-1,-1): #For each age 79 to 0

    #Create interpolation of next-period value function
    #Vp_interp = interp(np.log(x_grid),V[age],kind='linear',fill_value='extrapolate',bounds_error=False )
    Vp_interp = interp(np.log(x_grid),V[age],kind='cubic',fill_value=(np.min(V[age]),np.max(V[age])),bounds_error=False )

    for ix, x in enumerate(x_grid):  #For each cash-in-hand value
        print('Age',age+20,'and grid point',ix)
        
        bounds = [(0,1.0),(0,x.copy() )]
                
        if age >= 44:    #If retired
        
            Vp_value_temp = lambda z: Vp_Retired(x,z[0],z[1],p.survprob[age],Vp_interp)
            
            
            if age == T-1:
                inits = (.5,x/2) 
            else:
                inits = (A[age,ix],C[age,ix])
             #tol=.0000001
            solution = differential_evolution(Vp_value_temp,bounds,tol=.0000001,maxiter=1000000)
            
        if age < 44:    #If working
        
            Vp_value_temp = lambda z: Vp_Working(x,z[0],z[1],p.survprob[age],Vp_interp,age+20)

            inits = (A[age,ix],C[age,ix])
            solution = differential_evolution(Vp_value_temp,bounds,tol=.0000001,maxiter=1000000)

        print(solution.message)
        
        A[age-1,ix] = solution.x[0]
        C[age-1,ix] = solution.x[1]
        V[age-1,ix] = -Vp_value_temp(solution.x)

#------------------------------#
#                              #
#   Export Output and Plots    #
#                              # 
#------------------------------#

[plt.plot(A[78-i]) for i in range(78)]
plt.ylabel('Share of Risky Asset')
plt.xlabel('Wealth')
plt.title('Portfolio Decision in Working Period')
#plt.savefig('cubicretire1.png')


plt.close()
plt.figure(2)
plt.subplot(1,2,1)
[plt.plot(x_grid,A[78-i]) for i in range(78)]
plt.ylim(0,1)
plt.title('Portfolio Decision')
plt.subplot(1,2,2)
[plt.plot(x_grid,C[78-i]) for i in range(78)]
plt.title('Consumption Decision')

#import os
#os.chdir("/home/pando004/Desktop")
#np.savetxt("valuedc.csv",V,delimiter=",")
#np.savetxt("consumedc.csv",C,delimiter=",")
#np.savetxt("alphadc.csv",A,delimiter=",")



