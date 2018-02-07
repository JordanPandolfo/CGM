"""
Author: Jordan Pandolfo

Lifecycle Model Simulations with Fixed Income Savings
"""
#------------------------------------------------#
#                                                #
#   Import Packages, Functions and Parameters    #
#                                                #
#------------------------------------------------#
import numpy as np
#from beautifultable import BeautifulTable    
import matplotlib.pyplot as plt
from numpy import genfromtxt
from scipy.interpolate import interp2d as interp

#---------------------------------#
#                                 #
#   Import Parameters and Data    #
#                                 #
#---------------------------------#
import Lifecycle_data
p = Lifecycle_data 

delta    = 0.96
gamma    = 10
rf       = .02
mu       = .06    #mean risky asset return
eps_mu   = 0
sigr     = 0.157    #risky asset standard deviation
sige     = .07
rho      = 0
K        = 45    #Forced retirement age
T        = 80    #Max years of living, starting at working age 20

scale = 2.3
b0 = -2.170042+2.700381 #labor income regression intercept
b1 = 0.16818  #Age
b2 = -0.0323371/10 #Age**2/10
b3 = 0.0019704/100 #Age**3/100


def working(age,epsilon):
    """
    Inputs: 1) Age of worker
            2) Gross percentage shock to income
    
    Return: Labor income
    """
    
    return np.exp(b0+b1*age+b2*age**2+b3*age**3+epsilon)*scale  #*2 #1.5  #*scale  
    
Retire_Income = 25

#Grid points   
xa, xb, xn         = 7.5, 2000, 400
ya, yb, yn         = 7.5, 100, 100
x_grid     = np.linspace(xa,xb,xn)
y_grid      = np.linspace(ya,yb,yn)

mean    = np.array((mu,eps_mu))
cov     = np.array(( 
                     (sigr**2,rho),
                     (rho,sige**2)
                   ))


A = genfromtxt('alphaFS_income.csv',delimiter=',')
C = genfromtxt('consumeFS_income.csv',delimiter=',')
A = A.reshape(T,xn,yn)
C = C.reshape(T,xn,yn)


"""
added = 5
x_grid_aug = np.append(np.linspace(.5,7,added),x_grid).copy()

A_aug = np.ones((np.shape(A)[0],added,yn))
A_aug = np.append(A_aug,A,axis=1).copy()


C_aug= np.zeros((T,added,yn))

for age in range(T):
    print(age)
    for y in range(yn):
        for add in range(5):
            C_aug[age,add,y] = C[age,0,y]-.4*(x_grid[0]-np.linspace(.5,7,added)[add] )


C_aug = np.append(C_aug,C,axis=1).copy()
x_grid = x_grid_aug.copy()
A = A_aug.copy()
C = C_aug.copy()
"""

#-----------------------#
#                       #   
#   Policy Functions    #
#                       #
#-----------------------#

#-----------------------#
#                       #
#   Model Simulation    #
#                       #
#-----------------------#
N = 1000 #Number of simulations
W0 = 0 #initial wealth
Record = np.zeros((N,T,7))
death_date = np.zeros(N)

for sim in range(N):
    print(sim)
    wealth = W0
    Dead = 0
    Age = 0
    alpha = 1    
    
    while Dead == 0:
                
        if Age < 45: #if working
                    
            C_interp = interp(x_grid,y_grid,C[Age].T,kind='linear',bounds_error=False )
            alpha_interp = interp(x_grid,y_grid,A[Age].T,kind='cubic',bounds_error=False )
            
            #Labor income process            
            labor_income = working(Age+20,np.random.normal(eps_mu,sige,1))          

            #Wealth return process            
            r_shock = np.random.normal(mu,sigr,1)
            wealth_income = wealth*(alpha*(1+r_shock)+(1-alpha)*(1+rf))
            
            if wealth_income + labor_income < 15:
                consume = wealth_income + labor_income
                alpha   = 1
            else:
                consume = C_interp(wealth_income,labor_income)
                alpha   = alpha_interp(wealth_income,labor_income)
            
            #wealth_index = np.argmin(np.abs(x_grid-wealth_income))
            #labor_index  = np.argmin(np.abs(y_grid-labor_income))
            #consume = C[Age,wealth_index,labor_index]
            #alpha = A[Age,wealth_index,labor_index]
            
            wealth = wealth_income+labor_income - consume
            
            income_saving = labor_income - consume
            
            Record[sim,Age,0] = labor_income
            Record[sim,Age,1] = wealth_income
            Record[sim,Age,2] = wealth_income+labor_income
            Record[sim,Age,3] = consume
            Record[sim,Age,4] = alpha
            Record[sim,Age,5] = income_saving/labor_income
            Record[sim,Age,6] = Dead + 1
            
            #death roll
            Dead = np.random.binomial(1,1-p.survprob[Age])
            if Dead == 1:
                death_date[sim] = Age
            #Age update
            Age = Age + 1
                       
        elif Age >= 45: #if retired
            C_interp = interp(x_grid,y_grid,C[Age].T,kind='linear',bounds_error=False )
            alpha_interp = interp(x_grid,y_grid,A[Age].T,kind='cubic',bounds_error=False )
                        
            #Labor income process
            labor_income = Retire_Income

            #Wealth return process            
            r_shock = np.random.normal(mu,sigr,1)
            wealth_income = wealth*(alpha*(1+r_shock)+(1-alpha)*(1+rf))
 
            if wealth_income + labor_income < 15:
                consume = wealth_income+labor_income
                alpha   = 1
            else:
                consume = C_interp(wealth_income,labor_income)
                alpha   = alpha_interp(wealth_income,labor_income)
            
            #wealth_index = np.argmin(np.abs(x_grid-wealth_income))
            #labor_index  = np.argmin(np.abs(y_grid-labor_income))  
            #consume = C[Age,wealth_index,labor_index]          
            #alpha = A[Age,wealth_index,labor_index]
            
            wealth = wealth_income+labor_income - consume
            income_saving = labor_income - consume
            
            Record[sim,Age,0] = labor_income
            Record[sim,Age,1] = wealth_income
            Record[sim,Age,2] = wealth_income+labor_income
            Record[sim,Age,3] = consume
            Record[sim,Age,4] = alpha
            Record[sim,Age,5] = income_saving/labor_income
            Record[sim,Age,6] = Dead + 1
            
            #death roll
            Dead = np.random.binomial(1,1-p.survprob[Age])
            #Age update
            if Dead == 1:
                death_date[sim] = Age
                
            Age = Age + 1
            
            if Age == 80:
                Dead = 1

#--------------------------------------#
#                                      #
#   Computing Simulation Statistics    #
#                                      #    
#--------------------------------------#
laboravg = np.zeros(80)
wealthavg = np.zeros(80)
cashavg = np.zeros(80)
shareavg = np.zeros(80)
consumeavg = np.zeros(80)
labor_savingavg = np.zeros(80)

number_alive = np.zeros(80)

for Age in range(80): #for each age group
    #Reset compiler lists
    print(Age)
    labor        = []
    wealth       = []
    cash         = []
    consume      = []
    share        = []
    labor_saving = []    
    for sim in range(N): #for each sim
        if Record[sim,Age,6] == 1: #if sim still alive
            
            labor        = np.append(labor,Record[sim,Age,0])
            wealth       = np.append(wealth,Record[sim,Age,1])
            cash         = np.append(cash,Record[sim,Age,2])
            consume      = np.append(consume,Record[sim,Age,3])
            share        = np.append(share,Record[sim,Age,4])
            labor_saving = np.append(labor_saving,Record[sim,Age,5])
            
            number_alive[Age] = number_alive[Age] + 1
    #Compute average levels for age group     
    laboravg[Age]        = np.sum(labor)/number_alive[Age]
    wealthavg[Age]       = np.sum(wealth)/number_alive[Age]
    cashavg[Age]         = np.sum(cash)/number_alive[Age]
    shareavg[Age]        = np.sum(share)/number_alive[Age]
    consumeavg[Age]      = np.sum(consume)/number_alive[Age]
    labor_savingavg[Age] = np.sum(labor_saving)/number_alive[Age]


#-----------------------#
#                       #
#   Simluation Plots    #
#                       #
#-----------------------#

#Average Lifespan and mortality charts
life_expectancy = np.sum(death_date)/N+20
print('People live',life_expectancy,'years, on average.')
#Equity Share Glidepath
ages = np.linspace(20,100,80)
plt.close()
plt.figure(3)
plt.plot(ages[:70],shareavg[:70])
plt.title('Risky Asset Share Glidepath')
plt.ylim(0,1)
plt.xlabel('Age')
plt.ylabel('Fraction Invested in Risky Asset')
#plt.savefig('glide_no.png')

#Fraction of Labor Income Saved
aggregate_avg = (laboravg-consumeavg)/laboravg
plt.figure(4)
plt.plot(ages[0:45],100*labor_savingavg[0:45],label='individual')
#plt.plot(ages[0:45],100*aggregate_avg[0:45],label='aggregate')
plt.title('Fraction of Labor Income Saved \n Ages 20 to 65')
plt.ylabel('Fraction(%)')
plt.xlabel('Age')
plt.ylim(-20,45)
#plt.legend()
plt.axhline()
#plt.savefig('save_no.png')

#Wealth, Consumption and Labor Income
plt.figure(5)
plt.plot(ages,laboravg,label='Labor Income')
plt.plot(ages,consumeavg,label='Consumption')
plt.title('Income and Consumption')
plt.xlabel('Age')
plt.ylabel('Thousands of \$')
plt.axvline(x=65,color='k',linestyle='--',label='Retirement Age')
plt.legend()
#plt.savefig('wealth_no.png')


table = BeautifulTable()
table.column_headers = ["Object","Value","Units"]
table.append_row(["Avg Savings (Working Period)",np.mean(labor_savingavg[0:45]),'Decimal %' ])
table.append_row(["Avg Consumption (Last 5 Years Working)",np.mean(consumeavg[40:45]),'$,Thousands' ])
table.append_row(["Avg Consumption (Retirement Period)",np.mean(consumeavg[45:]) ,'$,Thousands'])
table.append_row(["Avg Income (Working Period)",np.mean(laboravg[:45]) ,'$,Thousands'])
table.append_row(["Median Income (Working Period)",np.median(laboravg[:45]) ,'$,Thousands'])
print(table)


plt.figure(6)
plt.plot(ages,cashavg,label='Cash-in-Hand')
plt.xlabel('Age')
plt.ylabel('Thousands of \$')
plt.axvline(x=65,color='k',linestyle='--',label='Retirement Age')
plt.title('Cash-in-Hand (CiH)')
plt.legend()
#plt.savefig('cih_no.png')


#Distribution of wealth and income at retirement, conditioned on
#   still being alive
labor = []
cash  = []
for person in range(N): #for each age group

    if Record[person,44,6] == 1: #if sim still alive
    
        labor = np.append(labor,Record[person,44,0])
        cash  = np.append(cash,Record[person,44,2])

plt.close()
plt.figure(7)
plt.subplot(1,2,1)
plt.hist(labor,bins=40,histtype='bar')
plt.title('Period-Before-Retirement Income \n from %s Simulations' %N)
plt.ylabel('Frequency')
plt.xlabel('Thousands of $')
plt.subplot(1,2,2)
plt.hist(cash,bins=40,histtype='bar')
plt.title('Wealth at Retirement \n from %s Simulations' %N)
plt.ylabel('Frequency')
plt.xlabel('Thousands of $')
#plt.savefig('dist_no.png')

table = BeautifulTable()
table.column_headers = ["Object","Value ($, Thounsads)"]
table.append_row(["Mean Retirement CiH",np.mean(cash) ])
table.append_row(["Median Retirement CiH",np.median(cash) ])
table.append_row(["10th Percentile CiH",np.percentile(cash,10)  ])
table.append_row(["25th Percentile CiH",np.percentile(cash,25)  ])
table.append_row(["75th Percentile CiH",np.percentile(cash,75)  ])
table.append_row(["90th Percentile CiH",np.percentile(cash,90)  ])
table.append_row(["SD Retirement CiH",np.std(cash)  ])
table.append_row(["Mean Age 64 Income",np.mean(labor)])
table.append_row(["SD Age 64 Income",np.std(labor)])
print(table)
