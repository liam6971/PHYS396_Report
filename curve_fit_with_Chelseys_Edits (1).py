#Importing all relevant packages
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np


#Function that only fits to T=1/(ax + b)
#this is a curve input used by the curve fitting algortihm 
def FractionOnly(x, a, b):
    return (1/(a*x + b))

#Function that fits to T=1/(px +(q/x)-r), where x=log R
#this is a curve input used by the curve fitting algortihm 
def FractionForLog(x, p, q, r):
    return 1/(p*x +(q/x)-r)

#Most of this code is Liam's but I put it inside a function wrapper so I could run it multiple times
#       curve: is the function that we are trying to fit
#       data: is a two element list of [[x axis data list], [y axis data list]]
#       guess: is a list containing the inital guesses for our coefficients
#       labels: is a two element list of [x axis string, y axis string]
def fit_curve_to(curve,data,guess,labels):
     
    popt, pcov = curve_fit(curve, data[0], data[1], guess)
    #Perform the curve-fit

    x_curve  = np.arange(np. min(data[0]), np. max(data[0]),0.01) 
    #generates x values for the curve.
    y_curve = curve(x_curve, *popt) 
    #generates y values for the curve, using constants. 

    plt.plot(data[0], data[1], 'bo', label='Measured Values')
    #Plots measured values

    #This generates the labels on the plot to 3 significant figures (see source referenced below)
    fittedLabel=""
    for i in range(len(guess)):
        popt_digits=-int(np.floor(np.log10(abs(popt[i])))) +2
        pcov_digits=-int(np.floor(np.log10(np.sqrt(abs(pcov[i][i])))))+2
        percenterror=round(pcov[i][i]/popt[i],6)*100
        fittedLabel+="\n"+str(round(popt[i],popt_digits))+"±"+str(round(np.sqrt(pcov[i][i]),pcov_digits))+"("+str(percenterror)+"%)"
    
    
    plt.plot(x_curve,y_curve,label=fittedLabel)
    #Plots the curve 

    print ("----Curve Fitter V 1.0----");
    print("Optimised Constants [a,c,k] =", popt)
    #prints values of a,b,k
    print ("-------------------------------");
    print("Covariance Matrix =\n",pcov)
    #prints 3x3 matrix whose main diagonal gives the variance of a,b,k 
    print ("-------------------------------");
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.legend()
    plt.show()


#Curve you want to fit as a function. 
x_data=[81052,81482,82050.9,81711.3,81429,81228.7,83220.9,83477.8,84257.7,81785,82042.2,82688.8,82938.8,83206,83397.8,83569,135940]

x_log_data = np.array([4.908763736, 4.91106168, 4.914083349, 4.91228212, 4.910779101, 4.909709503, 4.920232408, 4.921570995, 4.9256096, 4.912673658, 4.914037298, 4.917446689, 4.918757747, 4.920154644, 4.921154594, 4.922045205, 5.133347266]);
y_data = np.array([295.15, 288.15, 280.15, 285.15, 290.15, 292.15, 273.15, 270.15, 254.15, 284.15, 275.15, 268.15, 265.15, 263.15, 262.15, 261.15, 77.36]);

a_0 = 1.0
b_0 = 1.0
k_0 = 1.0

guess_ab = [a_0,b_0]
guess_abk = [0.3,-1,-2.9] 
#Initial guess for the parameters, both guesses stored in spreadsheet


NotLogData = [x_data,y_data]
LogData = [x_log_data,y_data]

fit_curve_to(FractionOnly,NotLogData,guess_ab,["R","T (K)"])

fit_curve_to(FractionForLog,LogData,guess_abk,["log(R)","T(K)"])


#Sources#
#https://www.youtube.com/watch?v=1H-SdMuJXTk
#https://www.youtube.com/watch?v=peBOquJ3fDo
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
#https://pubs.aip.org/aip/rsi/article/23/5/213/298007/The-Low-Temperature-Characteristics-of-Carbon
#https://stackoverflow.com/questions/25234996/getting-standard-error-associated-with-parameter-estimates-from-scipy-optimize-c

#Rounding to a certain significant figures: https://stackoverflow.com/questions/3410976/how-to-round-a-number-to-significant-figures-in-python
