# gain.py
import numpy as np
from numpy import random as rd
import matplotlib.pylab as plt
import time

start = time.time()

############### changable parameter ###############

t_exp = 60                              #exposure time(sec)
max_amp = 100                           #amplitude at the lowest frequency(ppm)
pattern = 'primary'                     #transit pattern ('primary' or 'secondary')


############### constant parameter ###############

h = 6.626070040*10**(-27)               #Plank constant(erg*s)
c = 2.99792458*10**10                   #speed of light(cm/s)
k = 1.38064852*10**(-16)                #Boltzmann constant(erg/K)
T_star = np.load('Data/temperature.npy')    #star temperature(K)
T_planet = 288.2                          #planet temperature(K)
R_Sun = 6.960*10**10                    #Solar radius(cm)
R_Earth = 6.378137*10**8                #Earth radius(cm)
R = np.load('Data/radius.npy')          #star radius(R_Sun)
R_star = R*R_Sun                        #star radius(cm)
R_planet = R_Earth                      #planet radius(cm)
p = R_planet/R_star                     #radius ratio
a = np.load('Data/semi.npy')            #semi major axis(cm)
P = np.load('Data/period.npy')          #period(sec)
P_tra = P*R_star/np.pi/a                #transit period(sec)
t_one = P_tra*3/t_exp                   #the number of plots in one transit observation
t_one = t_one.astype(np.int64)
t0 = [np.linspace(1, t_one[TT], t_one[TT]) for TT in range(len(T_star))]
lamb = np.load('Data/lamb.npy')         #wavelength(µm)
R1 = 100                                #resolution of detector1(6~11μm)
R2 = 100                                #resolution of detector2(11~18μm)
dl1 = (5+10)/2/R1                       #wavelength band(6~11µm)
dl2 = (10+20)/2/R2                      #wavelength band(11~18µm)
sci_pix_1 = 200*int(594*dl1/(10-5))     #the number of science pixels by wavelength on detector1
sci_pix_2 = 200*int(594*dl2/(20-10))    #the number of science pixels by wavelength on detector2
back_pix_1 = sci_pix_1                  #the number of background pixels by wavelength on detector1
back_pix_2 = sci_pix_2                  #the number of background pixels by wavelength on detector2
ref_pix = 1024*1024-200*594*2           #the number of reference pixels on detector1,2
dark_value = 0.2*t_exp                  #dark current value(electron)
offset_value = 14*np.sqrt(8*3/t_exp)    #offset value(electron)
n_gate = 4                              #the number of gates
zodi = np.load('Data/zodi.npy')         #zodiacal light by wavelength(electron)

def main():
    gain = np.array([gain_fluctuation(1, 0) for _ in range(4)])
    np.save('gain.npy', gain)

def gain_fluctuation(TT, ii):
    if ii==0:
        rd.seed()
    else:
        rd.seed(ii)
    period = np.linspace(60*2, t_one[TT]*60*4, 10000)
    frequency = 1/period
    alpha = np.array([rd.random() for _ in range(len(frequency))])
    rd.seed()
    fluctuation = np.sum([Fluctuation(f, a, t0[TT]*60)
                          for f, a in zip(frequency, alpha)], axis=0)*max_amp*10**(-6)*np.sqrt(frequency[-1])
    gain = (0.9+rd.random()/10)*(1+fluctuation)
    return gain[0]

def Fluctuation(f, a, t):
    return np.sin(2*np.pi*(f*t+a+rd.random()*np.exp(-1/f)))/np.sqrt(f)

if __name__=='__main__':
    main()
