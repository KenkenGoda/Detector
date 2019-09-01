#simulation_pr.py
import numpy as np
from scipy.stats import norm
from scipy.fftpack import fft
import random as rd
import matplotlib.pylab as plt
import time

start = time.time()

############### changable parameter ###############

n = 1                                   #repeat times
t_exp = 60                              #exposure time(sec)
times = 30                              #number of transmissions


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
t0 = [np.linspace(1,t_one[TT],t_one[TT]) for TT in range(len(T_star))]          #time for one transmission(min)
t = [np.linspace(1, t_total[TT], t_total[TT]) for TT in range(len(T_star))]     #time for total(min)
peak_to_peak = 500                      #peak-to-peak of gain fluctuation(ppm)
T_start = 1                             #the shortest period of gain fluctuation
T_end = t_one*4                         #the longest period of gain fluctuation
T_long = T_end-T_start+1
lamb = np.load('Data/lamb.npy')         #wavelength(µm)
R1 = 100                                #resolution of detector1(6~11μm)
R2 = 100                                #resolution of detector2(11~18μm)
dl1 = (5+11)/2/R1                       #wavelength band(6~11µm)
dl2 = (11+20)/2/R2                      #wavelength band(11~18µm)
sci_pix_1 = 200*int(500*dl1/(11-5))     #the number of science pixels by wavelength on detector1
sci_pix_2 = 200*int(500*dl2/(20-11))    #the number of science pixels by wavelength on detector2
back_pix_1 = sci_pix_1                  #the number of background pixels by wavelength on detector1
back_pix_2 = sci_pix_2                  #the number of background pixels by wavelength on detector2
ref_pix = 1024*1024-200*500*2           #the number of reference pixels on detector1,2
dark_value = 0.2*t_exp                  #dark current value(electron)
offset_value = 14*np.sqrt(8*3/t_exp)    #offset value(electron)
n_gate = 4                              #the number of gates
zodi = np.load('Data/zodi.npy')         #zodiacal light by wavelength(electron)


############### main ###############

def main():
    global sub_evaluation, sub_sigma
    sub_evaluation = np.array([[[0]*len(lamb)]*4]*n, float)
    sub_sigma = np.array([[[0]*len(lamb)]*4]*n, float)
    [[evaluation(nn, TT) for TT in range(4)] for nn in range(n)]
    np.save('Result/sci_evaluation_%d_%d_1.npy'%(times, n), sci_evaluation)
    np.save('Result/sci_sigma_%d_%d_1.npy'%(times, n), sci_sigma)
    print(sub_evaluation)
    print('Success')

############### make the gain fluctuation ###############

def gain_fluctuation(TT, jj):
    for tt in range(times):
        for PP in range(T_long[TT]):
            fluctuation = np.array([0]*t_one[TT], float)
            r = rd.random()
            fluctuation = np.array([peak_to_peak*4*10**(-8)*np.sin(2*np.pi/(PP+1)*kk+r*10)*(PP+1)/T_end[TT] for kk in range(t_one[TT])], float)
            if PP+1==T_start:
                gain_one = fluctuation
            else:
                gain_one += (-1)**rd.randint(1,2)*fluctuation
        gain_one += 0.9+rd.random()/10
        if tt==0:
            gain = gain_one
        else:
            gain = np.hstack((gain, gain_one))
    return gain


############### make the science curves ###############

def science(TT, jj, ll, gain, n_pix):
    tra = tra_one[ll]
    for tt in range(times-1):
        tra = np.hstack((tra, tra_one[ll]))
    x_star = (h*c)/(lamb[ll]*10**(-4)*k*T_star[TT])
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_star = abs(np.exp(x_star)/(np.exp(x_star)-1))
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_sci_ijk = np.array([[0]*t_total[TT]]*n_pix, float)
    D_sci_ijk = np.array([[rd.gauss((tra[kk]+zodi[ll])/(n_pix*n_gate)+dark_value, np.sqrt((bose_star*tra[kk]+bose_zodi*zodi[ll])/(n_pix*n_gate)+dark_value+offset_value)) for kk in range(t_total[TT])] for ii in range(n_pix)], float)*gain
    D_sci_ave_k = np.mean(D_sci_ijk, axis=0)
    return D_sci_ave_k


############### make the background curves ###############

def background(TT, ll, gain, n_pix):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_back_ijk = np.array([norm.rvs(zodi[ll]/(n_pix*n_gate)+dark_value, np.sqrt((bose_zodi*zodi[ll])/(n_pix*n_gate)+dark_value+offset_value), t_total[TT]) for ii in range(n_pix)], float)*gain
    D_back_ave_k = np.mean(D_back_ijk, axis=0)
    return D_back_ave_k


############### make the reference curves ###############

def reference(TT, gain, n_pix):
    D_ref_ijk = np.array([norm.rvs(dark_value, np.sqrt(dark_value+offset_value), t_total[TT]) for ii in range(n_pix)], float)*gain
    D_ref_ave_k = np.mean(D_ref_ijk, axis=0)
    return D_ref_ave_k


############### each average curve ###############

def unify(nn, TT):
    D_sci_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_back_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_ref_ave_k = np.array([[0]*t_total[TT]]*n_gate, float)
    for jj in range(n_gate):
        gain = gain_fluctuation(TT, jj)
        for ll in range(len(lamb)):
            if lamb[ll]<11:
                sci_pix = sci_pix_1
                back_pix = back_pix_1
            else:
                sci_pix = sci_pix_2
                back_pix = back_pix_2
            D_sci_ave_k[jj, ll] = science(TT, jj, ll, gain, int(sci_pix/n_gate))
            D_back_ave_k[jj, ll] = background(TT, ll, gain, int(back_pix/n_gate))
        D_ref_ave_k[jj] = reference(TT, gain, int(ref_pix/n_gate))
        elapsed_time = time.time()-start
        print('(n,T,j)=(%d,%d,%d): %dh%dm%ds'%(nn+1, T_star[TT], jj+1, int(elapsed_time/3600), int((elapsed_time-int(elapsed_time/3600)*3600)/60), elapsed_time-int(elapsed_time/60)*60))
    D_sci_ave = np.mean(D_sci_ave_k, axis=0)
    D_back_ave = np.mean(D_back_ave_k, axis=0)
    D_ref_ave = np.mean(D_ref_ave_k, axis=0)
    return D_sci_ave, D_back_ave, D_ref_ave


############### calibrate the gain fluctuation ###############

def gain_calibration(nn, TT):
    D_sci_ave, D_back_ave, D_ref_ave = unify(nn, TT)
    D_sub = np.array([[0]*t_total[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        for tt in range(times):
            sci_mean = 0
            back_mean = 0
            ref_mean = 0
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                sci_mean += D_sci_ave[ll, kk]
                back_mean += D_back_ave[ll, kk]
                ref_mean += D_ref_ave[kk]
            sci_mean /= t_one[TT]
            back_mean /= t_one[TT]
            ref_mean /= t_one[TT]
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                D_sub[ll, kk] = D_sci_ave[ll,kk]-(D_ref_ave[kk]-ref_mean)*sci_mean/ref_mean-back_mean
    return D_sub


############### evaluate this simulation ###############

def evaluation(nn, TT):
    global tra_one, z
    tra_one = np.load('Data/primary_%d.npy'%T_star[TT])
#    tra_one = np.load('Data/secondary_%d.npy'%T_star[TT])
    z = np.load('Data/parameter_%d.npy'%T_star[TT])
    D_sub = gain_calibration(nn, TT)
    for ll in range(len(lamb)):
        D_bin = np.array([0]*t_one[TT],float)
        for tt in range(times):
            for kk in range(t_one[TT]):
                D_bin[kk] += D_sub[ll,kk+tt*t_one[TT]]
        D_bin /= times
        top = 0
        n_top = 0
        for kk in range(t_one[TT]):
            if z[kk]>1+p[TT]:
                top += D_bin[kk]
                n_top += 1
        D_bin /= top/n_top
        D_mod_ave = average(TT, tra_one[ll])
        D_sub_ave = average(TT, D_sub[ll])
        D_mod_ave /= D_mod_ave[10]
        D_sub_ave /= D_sub_ave[10]
        sub_evaluation[nn, TT, ll] = (D_mod_ave[int(t_one[TT]/2)]-D_sub_ave[int(t_one[TT]/2)])*10**6
        sigma_top = []
        sigma_bottom = []
        for kk in range(t_one[TT]):
            if z[kk]>1+p[TT]:
                sigma_top.append(D_bin[kk])
            elif z[kk]<1-p[TT]:
                sigma_bottom.append(D_bin[kk])
        sigma_top = np.std(sigma_top)/np.sqrt(len(sigma_top))
        sigma_bottom = np.std(sigma_bottom)/np.sqrt(len(sigma_bottom))
        sub_sigma[nn, TT, ll] = np.sqrt(sigma_top**2+sigma_bottom**2)*10**6


############### option ###############

def average(TT,X):
    average = np.array([0]*t_one[TT], float)
    if len(X)==t_one[TT]:
        average = X
    else:
        for kk in range(t_one[TT]):
            for tt in range(times):
                average[kk] += X[kk+tt*t_one[TT]]
        average /= times
    top = 0
    bottom = 0
    n_top = 0
    n_bottom = 0
    for kk in range(t_one[TT]):
        if z[kk]>1+p[TT]:
            top += average[kk]
            n_top += 1
        elif z[kk]<1-p[TT]:
            bottom += average[kk]
            n_bottom += 1
    for kk in range(t_one[TT]):
        if z[kk]>1+p[TT]:
            average[kk] = top/n_top
        elif z[kk]<1-p[TT]:
            average[kk] = bottom/n_bottom
    return average


##############################

if __name__ == '__main__':
    main()

##############################
