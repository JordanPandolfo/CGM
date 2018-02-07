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
from scipy.interpolate import interp2d as interp
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

#save = genfromtxt('save.csv',delimiter=',')
save = np.ones(T)*0

#Grid points   
xa, xb, xn         = 7.5, 2000, 1000
ya, yb, yn         = 7.5, 100, 100
x_grid     = np.linspace(xa,xb,xn)
y_grid      = np.linspace(ya,yb,yn)

#Quadrature points and weights
ng, ne  = 10,10
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

#A = genfromtxt('alpha.csv',delimiter=',')
#C = genfromtxt('consume.csv',delimiter=',')
#V = genfromtxt('value.csv',delimiter=',')

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
    
    return np.exp(b0+b1*age+b2*age**2+b3*age**3+epsilon)*scale

def u(c):
    return c**(1-gamma)/(1-gamma)

"""
def Vp_Retired(x,y,a,c,pi,Interp):

    Inputs: 1) Cash-in-hand
            2) Portfolio share of risky asset
            3) Consumption
            4) Mortality probability
            5) Interpolating function
    Return: utility + discounted value function

    utility = u(c)
    
    discount = delta*pi
    
    expected_value = np.sum(quad_vector*Interp( np.log( (x+y-c)*(a*return_vector+(1-a)*(1+rf)  ) ), np.log(Retire_Income ) ) )
    
    return -(utility + discount*expected_value)    
"""

def Vp_Retired(x,y,a,c,pi,Interp):   
    return -(c**(1-gamma)/(1-gamma)+ delta*pi*np.sum(quad_vector*Interp( np.log( (x+y-c)*(a*return_vector+(1-a)*(1+rf)  ) ), np.log(Retire_Income ) ) ) )
"""
def Vp_Working(x,y,a,c,pi,Interp,age):

    Inputs: 1) Cash-in-hand
            2) Portfolio share of risky asset
            3) Consumption
            4) Mortality probability
            5) Interpolating function
            6) Age of worker 
    Return: utility + discounted value function

    utility = u(c)
    
    discount = delta*pi
    
    expected_value = np.sum(  
            joint*Interp( np.log((x+y-c)*(a*return_vector_working+(1-a)*(1+rf)  )), np.log(working(age,eps_vector)) ) 
            )
    
    return -(utility + discount*expected_value)
"""
def Vp_Working(x,y,a,c,pi,Interp,age):
    return -(c**(1-gamma)/(1-gamma) + delta*pi*np.sum(joint*Interp( np.log((x+y-c)*(a*return_vector_working+(1-a)*(1+rf)  )), np.log(working(age,eps_vector)) ) ))

#-----------------#
#                 #
#   Initialize    #
#                 #
#-----------------#
V = np.zeros((T,xn,yn))
A = np.zeros((T,xn,yn))
C = np.zeros((T,xn,yn))
V[T-1] = ( np.tile(x_grid,(yn,1)).T+np.tile(y_grid,(xn,1))   )**(1-gamma)/(1-gamma) 

#---------------------#
#                     #
#   Estimate Model    #
#                     #
#---------------------#
#V = genfromtxt('vtemp.csv',delimiter=',')
#A = genfromtxt('atemp.csv',delimiter=',')
#C = genfromtxt('ctemp.csv',delimiter=',')
#A = A.reshape(T,xn,yn)
#C = C.reshape(T,xn,yn)
#V = V.reshape(T,xn,yn)


for age in range(T-1,-1,-1): #For each age 79 to 0

    #Create interpolation of next-period value function
    Vp_interp = interp(np.log(x_grid),np.log(y_grid),V[age].T,kind='cubic',bounds_error=False )

    for ix, x in enumerate(x_grid):  #For each cash-in-hand value
        for iy, y in enumerate(y_grid): #For each labor income value                      
    
    
            print('Age',age+20,'and wealth',ix,'and income',iy)
                
            if age >= 44:    #If retired
            
                if age == T-1:
                    inits = (.5,x/2) 
                else:
                    inits = (A[age,ix],C[age,ix])
        
                bounds = [ (0,1.0),(0,x.copy()+y.copy() ) ]
        
                Vp_value_temp = lambda z: Vp_Retired(x,y,z[0],z[1],p.survprob[age],Vp_interp)
                
                solution = differential_evolution(Vp_value_temp,bounds,tol=.00001,maxiter=10000)    
                #tol=.0000001,maxiter=1000000                
            
            if age < 44:    #If working
            
                inits = (A[age,ix],C[age,ix])
                bounds = [ (0,1.0), (0,x.copy()+(1-save[age])*y.copy()) ]
        
                Vp_value_temp = lambda z: Vp_Working(x,y,z[0],z[1],p.survprob[age],Vp_interp,age+20)

                solution = differential_evolution(Vp_value_temp,bounds,tol=.00001,maxiter=10000)  
                #tol=.0000001,maxiter=1000000                  
        
            A[age-1,ix,iy] = solution.x[0]
            C[age-1,ix,iy] = solution.x[1]
            V[age-1,ix,iy] = -Vp_value_temp(solution.x)

#------------------------------#
#                              #
#   Export Output and Plots    #
#                              # 
#------------------------------#

#Same age,combos of (X,Y) where X+Y = W0
#Age 30 (190,10), (180,20),(170,30),(140,60)
#print(np.array(( C[10,37,3], C[10,35,14], C[10,33,25],C[10,27,56])) )
#Age 20 (10,190), (20,180),(30,170),(60,140)
#print(np.array(( C[0,37,3], C[0,35,14], C[0,33,25],C[0,27,56])) )

#import os
#os.chdir("/home/pando004/Desktop")
#np.savetxt("vtemp.csv",V.reshape(T,xn*yn),delimiter=",")
#np.savetxt("ctemp.csv",C.reshape(T,xn*yn),delimiter=",")
#np.savetxt("atemp.csv",A.reshape(T,xn*yn),delimiter=",")

