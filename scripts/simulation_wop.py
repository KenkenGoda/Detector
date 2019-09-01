#simulation_wop.py
import numpy as np
from numpy import random as rd
import matplotlib.pylab as plt
import time

start = time.time()

############### changable parameter ###############

t_exp = 60                              #exposure time(sec)
times = 30                              #number of transmissions
gain = 0.95
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
t_total = t_one*times                   #total times(2min)
t0 = [np.linspace(1, t_one[TT], t_one[TT]) for TT in range(len(T_star))]    #time for one transmission(min)
t = [np.linspace(1, t_total[TT], t_total[TT]) for TT in range(len(T_star))] #time for total(min)
peak_to_peak = 500                      #peak-to-peak of gain fluctuation(ppm)
T_start = 1                             #the shortest period of gain fluctuation
T_end = t_one*4                         #the longest period of gain fluctuation
T_long = T_end-T_start+1
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
    evaluation(1)
    print('Success')



############### make the science curves ###############

def science(TT, ll, n_pix):
    tra = tra_one[ll]
    for tt in range(times-1):
        tra = np.hstack((tra, tra_one[ll]))
    x_star = (h*c)/(lamb[ll]*10**(-4)*k*T_star[TT])
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_star = abs(np.exp(x_star)/(np.exp(x_star)-1))
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_sci_ijk = np.array([rd.normal((tra[kk]+zodi[ll])/(n_pix*n_gate)+dark_value,
                                     np.sqrt(((bose_star*tra[kk]+bose_zodi*zodi[ll])/(n_pix*n_gate)+dark_value+offset_value)/n_pix))
                          for kk in range(t_total[TT])], float)*gain
    return D_sci_ijk


############### make the background curves ###############

def background(TT, ll, n_pix):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_back_ijk = rd.normal(zodi[ll]/(n_pix*n_gate)+dark_value,
                           np.sqrt(((bose_zodi*zodi[ll])/(n_pix*n_gate)+dark_value+offset_value)/n_pix), t_total[TT])*gain
    return D_back_ijk



############### each average curve ###############

def unify(TT):
    D_sci_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_back_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    for jj in range(n_gate):
        for ll in range(len(lamb)):
            if lamb[ll]<11:
                sci_pix = sci_pix_1
                back_pix = back_pix_1
            else:
                sci_pix = sci_pix_2
                back_pix = back_pix_2
            D_sci_ave_k[jj, ll] = science(TT, ll, int(sci_pix/n_gate))
            D_back_ave_k[jj, ll] = background(TT, ll, int(back_pix/n_gate))
        elapsed_time = time.time()-start
        print('(T,j)=(%d,%d): %dh%dm%ds'%(T_star[TT], jj+1, int(elapsed_time/3600),
                                          int((elapsed_time-int(elapsed_time/3600)*3600)/60),
                                          elapsed_time-int(elapsed_time/60)*60))
    D_sci_ave = np.mean(D_sci_ave_k, axis=0)
    D_back_ave = np.mean(D_back_ave_k, axis=0)
    return D_sci_ave, D_back_ave


############### calibrate the gain fluctuation ###############

def gain_calibration(TT):
    D_sci_ave, D_back_ave = unify(TT)
    D_sci = np.array([[0]*t_total[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        for tt in range(times):
            back_mean = np.mean([D_back_ave[ll, kk] for kk in range(tt*t_one[TT], (tt+1)*t_one[TT])])
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                D_sci[ll, kk] = D_sci_ave[ll, kk]-back_mean
    return D_sci


############### evaluate this simulation ###############

def evaluation(TT):
    global tra_one, z
    tra_one = np.load('Data/%s_%d.npy'%(pattern, T_star[TT]))
    D_sci = gain_calibration(TT)
    D_sci_norm = []
    for ll in range(len(lamb)):
        D_sci_norm.append([normalization(TT, tt, D_sci[ll]) for tt in range(times)])
    np.save('wop_norm_%s_%d.npy'%(pattern, T_star[TT]), D_sci_norm)

        
############### option ###############

def normalization(TT, tt, data):
    z = np.load('Data/parameter_%d.npy'%T_star[TT])
    D_norm = [data[kk+tt*t_one[TT]] for kk in range(t_one[TT])]
    top = np.mean([D_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]])
    D_norm /= top
    return D_norm


##############################

if __name__ == '__main__':
    main()

##############################
