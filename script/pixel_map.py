#pixel_map.py
import numpy as np
from numpy import random as rd
import matplotlib.pylab as plt
import time

start = time.time()

############### changable parameter ###############

j_start = 612
j_width = 200
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


############### main ###############

def main():
    for TT in range(1, 2):
        print(T_star[TT])
        transit = np.load('Data/%s_%d.npy'%(pattern, T_star[TT]))
        gain = np.load('gain.npy')
        data = np.array([[0]*j_width]*1024, float)
        for jj in range(j_start, j_start+j_width):
            print(jj)
            g = gain_select(jj, gain)
            if 212<=jj<412:
                for ii in range(1024):
                    if 215<=ii<809:
                        for ll in range(len(lamb)):
                            if 215+9*ll<=ii<215+9*(ll+1):
                                data[ii, jj-j_start] = science(TT, ll, transit, g*gain_fluctuation(TT, 1))
                    else:
                        data[ii, jj-j_start] = reference(TT, g*gain_fluctuation(TT, 1))
            elif 612<=jj<812:
                for ii in range(1024):
                    if 215<=ii<809:
                        for ll in range(len(lamb)):
                            if 215+9*ll<=ii<215+9*(ll+1):
                                data[ii, jj-j_start] = background(TT, ll, g*gain_fluctuation(TT, 1))
                    else:
                        data[ii, jj-j_start] = reference(TT, g*gain_fluctuation(TT, 1))
            else:
                for ii in range(1024):
                    data[ii, jj-j_start] = reference(TT, g*gain_fluctuation(TT, 1))
        np.save('pixel_map_%d_4.npy'%T_star[TT], data)
    print('Success')


############### make the gain fluctuation ###############

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


############### select the gain ###############

def gain_select(jj, gain):
    if jj%4==0:
        return gain[0]
    elif jj%4==1:
        return gain[1]
    elif jj%4==2:
        return gain[2]
    else:
        return gain[3]


############### make the science curves ###############

def science(TT, ll, transit, gain):
    tra = transit[ll]
    x_star = (h*c)/(lamb[ll]*10**(-4)*k*T_star[TT])
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_star = abs(np.exp(x_star)/(np.exp(x_star)-1))
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    return rd.normal((tra[0]+zodi[ll])/sci_pix_1+dark_value+offset_value,
                     np.sqrt((bose_star*tra[0]+bose_zodi*zodi[ll])/sci_pix_1+dark_value+offset_value))*gain


############### make the background curves ###############

def background(TT, ll, gain):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    return rd.normal(zodi[ll]/back_pix_1+dark_value+offset_value,
                     np.sqrt((bose_zodi*zodi[ll])/back_pix_1+dark_value+offset_value))*gain


############### make the reference curves ###############

def reference(TT, gain):
    return rd.normal(dark_value+offset_value, np.sqrt(dark_value+offset_value))*gain


##############################

if __name__ == '__main__':
    main()

##############################
