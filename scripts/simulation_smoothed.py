#simulation_smoothed.py
import numpy as np
from numpy import random as rd
from scipy.stats import norm
import matplotlib.pylab as plt
import time

start = time.time()

############### changable parameter ###############

j = 4
t_exp = 60                              # exposure time(sec)
times = 1                               # number of transmissions
Q = 1.0*t_exp                           # dark current value(electron)
gain_std = 100                          # standard deviation of gain(ppm)
pattern = 'primary'                     # transit pattern ('primary' or 'secondary')


############### constant parameter ###############

h = 6.626070040*10**(-27)               # Plank constant(erg*s)
c = 2.99792458*10**10                   # speed of light(cm/s)
k = 1.38064852*10**(-16)                # Boltzmann constant(erg/K)
T_star = np.load('Data/temperature.npy')  # star temperature(K)
T_planet = 288.2                          # planet temperature(K)
R_Sun = 6.960*10**10                    # Solar radius(cm)
R_Earth = 6.378137*10**8                # Earth radius(cm)
R = np.load('Data/radius.npy')          # star radius(R_Sun)
R_star = R*R_Sun                        # star radius(cm)
R_planet = R_Earth                      # planet radius(cm)
p = R_planet/R_star                     # radius ratio
a = np.load('Data/semi.npy')            # semi major axis(cm)
P = np.load('Data/period.npy')          # period(sec)
P_tra = P*R_star/np.pi/a                # transit period(sec)
t_one = P_tra*3/t_exp                   # the number of plots in one transit observation
t_one = t_one.astype(np.int64)
t_total = t_one*times                   # total times(2min)
t0 = [np.linspace(1, t_one[TT], t_one[TT]) for TT in range(len(T_star))]    # time for one transmission(min)
t = [np.linspace(1, t_total[TT], t_total[TT]) for TT in range(len(T_star))] # time for total(min)
lamb_bord = np.load('Data/lamb_bord.npy')   # wavelength borders(µm)
dl = np.load('Data/dl.npy')             # wavelength bands(µm)
lamb = np.load('Data/lamb.npy')         # wavelength(µm)
sci_pix_1 = 200*int(600*dl[0]/(lamb_bord[1]-lamb_bord[0]))  # the number of science pixels by wavelength on detector1
sci_pix_2 = 200*int(600*dl[1]/(lamb_bord[2]-lamb_bord[1]))  # the number of science pixels by wavelength on detector2
sci_pix_3 = 200*int(600*dl[2]/(lamb_bord[3]-lamb_bord[2]))  # the number of science pixels by wavelength on detector3
back_pix_1 = sci_pix_1                  # the number of background pixels by wavelength on detector1
back_pix_2 = sci_pix_2                  # the number of background pixels by wavelength on detector2
back_pix_3 = sci_pix_3                  # the number of background pixels by wavelength on detector3
ref_pix = 1024*1024-200*600*2           # the number of reference pixels on detector1, 2 and 3
fowler = 16                             # number of fowler sampling
read = 21.9/np.sqrt(fowler)             # readout noise(electron/read)
n_gate = 4                              # the number of gates
zodi = np.load('Data/zodi.npy')         # zodiacal light by wavelength(electron)


############### main ###############

def main():
#    [save(1, j+1) for j in range(4)]
    save(1, j)
    print('Success')


############### make the gain fluctuation ###############

def gain_fluctuation(TT, f_high, f_low, gain_com):
    gain_rand = gain_random(TT, f_high, f_low)
    fluctuation = gain_com+gain_rand
    return rd.normal(1, 0.00015)+fluctuation*(gain_std*10**(-6)/fluctuation.std())

def gain_common(TT, f_low):
    gain_com = np.sum([Fluctuation(f, t0[TT]*60) for f in f_low], axis=0)*np.sqrt(f_low[-1])
    return rd.normal(1, 0.00015)+gain_com*(gain_std*10**(-6)/gain_com.std())

def gain_random(TT, f_high, f_low):
    return np.sum([Fluctuation(f, t0[TT]*60) for f in f_high], axis=0)*np.sqrt(f_low[-1])

def Fluctuation(f, t):
    return np.sin(2*np.pi*(f*t+rd.random()))/np.sqrt(f)


############### make the science curves ###############

def science(TT, ll, gain, n_pix):
    tra = np.load('Data/%s_%d.npy'%(pattern, T_star[TT]))[ll]
    for tt in range(times-1):
        tra = np.hstack((tra, tra_one[ll]))
    x_star = (h*c)/(lamb[ll]*10**(-4)*k*T_star[TT])
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_star = abs(np.exp(x_star)/(np.exp(x_star)-1))
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
#    D_sci_ijk = np.array([rd.normal((tra[kk]+zodi[ll])/n_pix+Q,
#                                   np.sqrt((bose_star*tra[kk]+bose_zodi*zodi[ll])/n_pix+Q+read))
#                          for kk in range(t_total[TT])], float)*gain
    D_sci_ijk = norm.rvs((tra[kk]+zodi[ll])/n_pix+Q, np.sqrt(Q+read), t_total[TT])*gain
#    D_sci_ijk = ((tra+zodi[ll])/n_pix+Q)*gain
    return D_sci_ijk


############### make the background curves ###############

def background(TT, ll, gain, n_pix):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
#    D_back_ijk = norm.rvs(zodi[ll]/n_pix+Q,
#                          np.sqrt(bose_zodi*zodi[ll]/n_pix+Q+read), t_total[TT])*gain
    D_back_ijk = norm.rvs(zodi[ll]/n_pix+Q, np.sqrt(Q+read), t_total[TT])*gain
#    D_back_ijk = (zodi[ll]/n_pix+Q)*gain
    return D_back_ijk


############### make the reference curves ###############

def reference(TT, gain):
    D_ref_ijk = norm.rvs(Q, np.sqrt(Q+read), t_total[TT])*gain
#    D_ref_ijk = Q*gain
    return D_ref_ijk


############### each average curve ###############

def save(TT, jj):
    print('j=%d'%jj)
    P_short = np.linspace(1, 60*2, 10000)
    P_long = np.linspace(60*2, t_one[TT]*60*4, 10000)
    f_high = 1/P_short
    f_low = 1/P_long
    gain_com = gain_common(TT, f_low)
    lamb_10 = np.abs(lamb-10)
    ll = np.where(lamb_10==lamb_10.min())[0][0]
    sci_pix = sci_pix_2
    back_pix = back_pix_2
    D_sci_ijk = np.array([science(TT, ll, gain_com, sci_pix)
                          for _ in range(int(sci_pix/n_gate))])
    print('Finish science')
    D_back_ijk = np.array([background(TT, ll, gain_com, back_pix)
                          for _ in range(int(back_pix/n_gate))])
    print('Finish background')
    D_ref_ijk = np.array([reference(TT, gain_com)
                          for _ in range(int(ref_pix/n_gate))])
    print('Finish reference')
    elapsed_time = time.time()-start
    print('j=%d: %dh%dm%ds'%(jj, int(elapsed_time/3600),
                             int((elapsed_time-int(elapsed_time/3600)*3600)/60),
                             elapsed_time-int(elapsed_time/60)*60))
    np.save('Result/D_sci_%d.npy'%jj, D_sci_ijk)
    np.save('Result/D_back_%d.npy'%jj, D_back_ijk)
    np.save('Result/D_ref_%d.npy'%jj, D_ref_ijk)


##############################

if __name__ == '__main__':
    main()

##############################
