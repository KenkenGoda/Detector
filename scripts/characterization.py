#characterization.py
import numpy as np

T = np.array([2500,3000,3500,4000],float)  #star temperature(K)
T_planet = 288.2                                #planet temperature(K)
R = -8.133+5.09342*10**(-3)*T-9.86602*10**(-7)*T**2+6.47963*10**(-11)*T**3    #star radius(R_Sun)
acc = 8/7*10**(-2)              #proportionality constant
R[0] = 0.1                      #predicted by Trappist-1
R[1] = 0.1+acc*5                #predicted by fitting
R_Sun = 6.957*10**10            #the Sun radius(cm)
R_Earth = 6.371*10**8           #the Earth radius(cm)
A = 0.306                       #albedo of the Earth
a = R*R_Sun/2*(T/T_planet)**2*np.sqrt(1-A)       #semi major axis(cm)
G = 6.67408*10**(-5)            #gravitation constance(cm^3/kg/sec^2)
M_Sun = 1.989*10**30            #the Sun mass(kg)
M = (-0.6063+np.sqrt(0.6063**2-4*0.3200*(0.0906-R)))/0.64*M_Sun     #the star mass(kg)
P = 2*np.pi*np.sqrt(a**3/G/M)   #period(sec)
AU = 14959787070000             #1AU

print('T=', T, 'K')
print('R=', R, 'cm')
print('a=', a/AU, 'AU')
print('M=', M, 'kg')
print('P=', P, 'sec')

np.save('Data/temperature.npy',T)
np.save('Data/radius.npy',R)
np.save('Data/semi.npy',a)
np.save('Data/period.npy',P)

