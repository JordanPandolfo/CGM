import numpy as np
import matplotlib.pyplot as plt

#Parameters
delta = 0.96
gamma = 10
rf = .02
mu = .06   #mean risky asset return
sigr = 0.157 #risky asset standard deviation
K = 45 #Forced retirement age
T = 80 #Max years of living, starting at working age 20
b = 0 #bequest motive
sige = np.sqrt(.0738)
sigu = np.sqrt(.0106)
rho = 0
lamda = 0.68212 #replacement ratio
b0 = -2.170042+2.700381 #labor income regression intercept
b1 = 0.16818  #Age
b2 = -0.0323371/10 #Age**2/10
b3 = 0.0019704/100  #Age**3/100

cpi = 1.79 #Adjust from 1992 to 2017 dollars

#Conditional Probability of dying, given age (starting at 20)
#   2012 tables from National Center for Health Statistics
survprob = np.zeros((80,1))
survprob[0,0]= 0.000726    #At 20, prbbaility of alive at 21
survprob[1,0]= 0.000806
survprob[2,0]= 0.000868
survprob[3,0]= 0.000905
survprob[4,0]= 0.000924
survprob[5,0]= 0.000939
survprob[6,0]= 0.000957
survprob[7,0]= 0.000977
survprob[8,0]= 0.001002
survprob[9,0]= 0.001030
survprob[10,0]= 0.001061
survprob[11,0]= 0.001094
survprob[12,0]= 0.001127
survprob[13,0]= 0.001161
survprob[14,0]= 0.001200
survprob[15,0]= 0.001251
survprob[16,0]= 0.001317
survprob[17,0]= 0.001394
survprob[18,0]= 0.001480
survprob[19,0]= 0.001575
survprob[20,0]= 0.001680
survprob[21,0]= 0.001802
survprob[22,0]= 0.001950
survprob[23,0]= 0.002130
survprob[24,0]= 0.002344
survprob[25,0]= 0.002575
survprob[26,0]= 0.002824
survprob[27,0]= 0.003112
survprob[28,0]= 0.003437
survprob[29,0]= 0.003787
survprob[30,0]= 0.004146
survprob[31,0]= 0.004509
survprob[32,0]= 0.004884
survprob[33,0]= 0.005282
survprob[34,0]= 0.005708
survprob[35,0]= 0.006167
survprob[36,0]= 0.006651
survprob[37,0]= 0.007156
survprob[38,0]= 0.007673
survprob[39,0]= 0.008210
survprob[40,0]= 0.008784
survprob[41,0]= 0.009408
survprob[42,0]= 0.010083
survprob[43,0]= 0.010819
survprob[44,0]= 0.011628
survprob[45,0]= 0.012530
survprob[46,0]= 0.013534
survprob[47,0]= 0.014658
survprob[48,0]= 0.015888
survprob[49,0]= 0.017236
survprob[50,0]= 0.018831
survprob[51,0]= 0.020693
survprob[52,0]= 0.022723
survprob[53,0]= 0.024884
survprob[54,0]= 0.027216
survprob[55,0]= 0.029822
survprob[56,0]= 0.032876
survprob[57,0]= 0.036328
survprob[58,0]= 0.040156
survprob[59,0]= 0.044699
survprob[60,0]= 0.049419
survprob[61,0]= 0.054529
survprob[62,0]= 0.060341
survprob[63,0]= 0.067163
survprob[64,0]= 0.074785
survprob[65,0]= 0.083577
survprob[66,0]= 0.093319
survprob[67,0]= 0.103993
survprob[68,0]= 0.115643
survprob[69,0]= 0.128300
survprob[70,0]= 0.141986
survprob[71,0]= 0.156706
survprob[72,0]= 0.172451
survprob[73,0]= 0.189191
survprob[74,0]= 0.206875
survprob[75,0]= 0.225433
survprob[76,0]= 0.244768
survprob[77,0]= 0.264767
survprob[78,0]= 0.285296
survprob[79,0]= 0.306203     #At 99, probability of alive at 100

survprob = np.repeat(1,80)-survprob


#Quadrature weights and nodes
w = [.16666666666,.66666666666,.16666666666]
node = [-1.73205080756887, 0.0, 1.73205080756887]

def regress(age):
    #Deterministic income process
    return np.exp(b0+b1*age+b2*age**2+b3*age**3)*cpi

def u(c,a):
    #return np.log(c)
    return np.tile( c**(1-gamma)/(1-gamma),(len(a),1)).T

#Define grid space and policy functions
alphaa, alphab = 0.0, 1.0
ca, cb = 0.0, 200
xa, xb = 0.0, 300
alphan, cn, xn = 30, 30, 30 #Number of grid points



"""
From original paper
#Conditional Survival Probabilities
survprob = np.zeros((80,1))
survprob[1-1,1-1] = 0.99845
survprob[2-1,1-1] = 0.99839
survprob[3-1,1-1] = 0.99833
survprob[4-1,1-1] = 0.9983
survprob[5-1,1-1] = 0.99827
survprob[6-1,1-1] = 0.99826
survprob[7-1,1-1] = 0.99824
survprob[8-1,1-1] = 0.9982
survprob[9-1,1-1] = 0.99813
survprob[10-1,0] = 0.99804
survprob[11-1,0] = 0.99795
survprob[12-1,1-1] = 0.99785
survprob[13-1,1-1] = 0.99776
survprob[14-1,1-1] = 0.99766
survprob[15-1,1-1] = 0.99755
survprob[16-1,1-1] = 0.99743
survprob[17-1,1-1] = 0.9973
survprob[18-1,1-1] = 0.99718
survprob[19-1,1-1] = 0.99707
survprob[20-1,1-1] = 0.99696
survprob[21-1,1-1] = 0.99685
survprob[22-1,1-1] = 0.99672
survprob[23-1,1-1] = 0.99656
survprob[24-1,1-1] = 0.99635
survprob[25-1,1-1] = 0.9961
survprob[26-1,1-1] = 0.99579
survprob[27-1,1-1] = 0.99543
survprob[28-1,1-1] = 0.99504
survprob[29-1,1-1] = 0.99463
survprob[30-1,1-1] = 0.9942
survprob[31-1,1-1] = 0.9937
survprob[32-1,1-1] = 0.99311
survprob[33-1,1-1] = 0.99245
survprob[34-1,1-1] = 0.99172
survprob[35-1,1-1] = 0.99091
survprob[36-1,1-1] = 0.99005
survprob[37-1,1-1] = 0.98911
survprob[38-1,1-1] = 0.98803
survprob[39-1,1-1] = 0.9868
survprob[40-1,1-1] = 0.98545
survprob[41-1,1-1] = 0.98409
survprob[42-1,1-1] = 0.9827
survprob[43-1,1-1] = 0.98123
survprob[44-1,1-1] = 0.97961
survprob[45-1,1-1] = 0.97786
survprob[46-1,1-1] = 0.97603
survprob[47-1,1-1] = 0.97414
survprob[48-1,1-1] = 0.97207
survprob[49-1,1-1] = 0.9697
survprob[50-1,1-1] = 0.96699
survprob[51-1,1-1] = 0.96393
survprob[52-1,1-1] = 0.96055
survprob[53-1,1-1] = 0.9569
survprob[54-1,1-1] = 0.9531
survprob[55-1,1-1] = 0.94921
survprob[56-1,1-1] = 0.94508
survprob[57-1,1-1] = 0.94057
survprob[58-1,1-1] = 0.9357
survprob[59-1,1-1] = 0.93031
survprob[60-1,1-1] = 0.92424
survprob[61-1,1-1] = 0.91717
survprob[62-1,1-1] = 0.90922
survprob[63-1,1-1] = 0.90089
survprob[64-1,1-1] = 0.89282
survprob[65-1,1-1] = 0.88503
survprob[66-1,1-1] = 0.87622
survprob[67-1,1-1] = 0.86576
survprob[68-1,1-1] = 0.8544
survprob[69-1,1-1] = 0.8423
survprob[70-1,1-1] = 0.82942
survprob[71-1,1-1] = 0.8154
survprob[72-1,1-1] = 0.80002
survprob[73-1,1-1] = 0.78404
survprob[74-1,1-1] = 0.76842
survprob[75-1,1-1] = 0.75382
survprob[76-1,1-1] = 0.73996
survprob[77-1,1-1] = 0.72464
survprob[78-1,1-1] = 0.71057
survprob[79-1,1-1] = 0.6961
survprob[80-1,1-1] = 0.6809
"""
