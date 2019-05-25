#simulation_wp.py
import numpy as np
from numpy import random as rd
import matplotlib.pylab as plt
import time
from datetime import datetime

start = time.time()

############### changable parameter ###############

volume = 4
n = 100                                 #repeat times
t_exp = 60                              #exposure time(sec)
times = 60                              #number of transmissions
dark_value = 5.0*t_exp                  #dark current value(electron)
pattern = 'secondary'                     #transit pattern ('primary' or 'secondary')


############### constant parameter ###############

h = 6.626070040*10**(-27)               #Plank constant(erg*s)
c = 2.99792458*10**10                   #speed of light(cm/s)
k = 1.38064852*10**(-16)                #Boltzmann constant(erg/K)
T_star = np.load('Data/temperature.npy')  #star temperature(K)
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
lamb = np.load('Data/lamb.npy')         #wavelength(µm)
R1 = 100                                #resolution of detector1(5~10μm)
R2 = 100                                #resolution of detector2(10~20μm)
dl1 = (5+10)/2/R1                       #wavelength band(6~11µm)
dl2 = (10+20)/2/R2                      #wavelength band(11~18µm)
sci_pix_1 = 200*int(594*dl1/(10-5))     #the number of science pixels by wavelength on detector1
sci_pix_2 = 200*int(594*dl2/(20-10))    #the number of science pixels by wavelength on detector2
back_pix_1 = sci_pix_1                  #the number of background pixels by wavelength on detector1
back_pix_2 = sci_pix_2                  #the number of background pixels by wavelength on detector2
ref_pix = 1024*1024-200*594*2           #the number of reference pixels on detector1,2
offset_value = 14*np.sqrt(8*3/t_exp)    #offset value(electron)
n_gate = 4                              #the number of gates
max_amp = 100                           #amplitude at lowest frequency
zodi = np.load('Data/zodi.npy')         #zodiacal light by wavelength(electron)


############### main ###############

def main():
    global sci_evaluation, sci_sigma, sub_evaluation, sub_sigma
    sci_evaluation = np.array([[[0]*len(lamb)]*4]*n, float)
    sub_evaluation = np.array([[[0]*len(lamb)]*4]*n, float)
    sci_sigma = np.array([[[0]*len(lamb)]*4]*n, float)
    sub_sigma = np.array([[[0]*len(lamb)]*4]*n, float)
    [evaluation(nn, TT) for nn in range(n) for TT in range(len(T_star))]
    #np.save('Result/wog_evaluation_{}_{}.npy'.format(pattern, volume), sci_evaluation)
    np.save('Result/sci_evaluation_{}_{}.npy'.format(pattern, volume), sci_evaluation)
    np.save('Result/sub_evaluation_{}_{}.npy'.format(pattern, volume), sub_evaluation)
    #np.save('Result/wog_sigma_{}_{}.npy'.format(pattern, volume), sci_sigma)
    np.save('Result/sci_sigma_{}_{}.npy'.format(pattern, volume), sci_sigma)
    np.save('Result/sub_sigma_{}_{}.npy'.format(pattern, volume), sub_sigma)
    print('Success')

############### make the gain fluctuation ###############

def gain_fluctuation(TT):
    period = np.linspace(60*2, t_one[TT]*60*4, 10000)
    frequency = 1/period
    for tt in range(times):
        fluctuation = np.sum([Fluctuation(f, t0[TT]*60) for f in frequency], axis=0)*max_amp*10**(-6)*np.sqrt(frequency[-1])
        gain_one = (0.9+rd.random()/10)*(1+fluctuation)
        if tt==0:
            gain = gain_one
        else:
            gain = np.hstack((gain, gain_one))
    return gain

def Fluctuation(f, t):
    return np.sin(2*np.pi*(f*t+rd.random()))/np.sqrt(f)


############### make the science curves ###############

def science(TT, ll, gain, n_pix):
    tra = tra_one[ll]
    for tt in range(times-1):
        tra = np.hstack((tra, tra_one[ll]))
    x_star = (h*c)/(lamb[ll]*10**(-4)*k*T_star[TT])
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_star = abs(np.exp(x_star)/(np.exp(x_star)-1))
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_sci_ijk = np.array([rd.normal((tra[kk]+zodi[ll])/n_pix+dark_value,
                                   np.sqrt(((bose_star*tra[kk]+bose_zodi*zodi[ll])/n_pix+dark_value+offset_value)/(n_pix/n_gate)))
                          for kk in range(t_total[TT])], float)*gain
    return D_sci_ijk


############### make the background curves ###############

def background(TT, ll, gain, n_pix):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_back_ijk = rd.normal(zodi[ll]/n_pix+dark_value,
                           np.sqrt(((bose_zodi*zodi[ll])/n_pix+dark_value+offset_value)/(n_pix/n_gate)), t_total[TT])*gain
    return D_back_ijk


############### make the reference curves ###############

def reference(TT, gain, n_pix):
    D_ref_ijk = rd.normal(dark_value, np.sqrt((dark_value+offset_value)/(n_pix/n_gate)), t_total[TT])*gain
    return D_ref_ijk


############### each average curve ###############

def unify(nn, TT):
    D_sci_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_back_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_ref_ave_k = np.array([[[0]*t_total[TT]]*n_gate]*2, float)
    for jj in range(n_gate):
        #gain = 1.0
        gain_1 = gain_fluctuation(TT)
        gain_2 = gain_fluctuation(TT)
        for ll in range(len(lamb)):
            if lamb[ll]<10:
                sci_pix = sci_pix_1
                back_pix = back_pix_1
                gain = gain_1
            else:
                sci_pix = sci_pix_2
                back_pix = back_pix_2
                gain = gain_2
            D_sci_ave_k[jj, ll] = science(TT, ll, gain, sci_pix)
            D_back_ave_k[jj, ll] = background(TT, ll, gain, back_pix)
        #D_ref_ave_k[0, jj] = reference(TT, gain, ref_pix)
        #D_ref_ave_k[1, jj] = reference(TT, gain, ref_pix)
        D_ref_ave_k[0, jj] = reference(TT, gain_1, ref_pix)
        D_ref_ave_k[1, jj] = reference(TT, gain_2, ref_pix)
    elapsed_time = time.time()-start
    print('(n,T)=(%d,%d): %dh%dm%ds'%(nn+1, T_star[TT], int(elapsed_time/3600),
                                      int((elapsed_time-int(elapsed_time/3600)*3600)/60),
                                      elapsed_time-int(elapsed_time/60)*60))
    D_sci_ave = np.mean(D_sci_ave_k, axis=0)
    D_back_ave = np.mean(D_back_ave_k, axis=0)
    D_ref_ave = np.mean(D_ref_ave_k, axis=1)
    return D_sci_ave, D_back_ave, D_ref_ave


############### calibrate the gain fluctuation ###############

def gain_calibration(nn, TT):
    D_sci_ave, D_back_ave, D_ref_ave = unify(nn, TT)
    D_sci = np.array([[0]*t_total[TT]]*len(lamb), float)
    D_sub = np.array([[0]*t_total[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        if lamb[ll]<10:
            D_ref = D_ref_ave[0]
        else:
            D_ref = D_ref_ave[1]
        for tt in range(times):
            sci_mean = 0
            back_mean = 0
            ref_mean = 0
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                sci_mean += D_sci_ave[ll, kk]
                back_mean += D_back_ave[ll, kk]
                ref_mean += D_ref[kk]
            sci_mean /= t_one[TT]
            back_mean /= t_one[TT]
            ref_mean /= t_one[TT]
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                D_sci[ll, kk] = D_sci_ave[ll, kk]-back_mean
                D_sub[ll, kk] = D_sci_ave[ll, kk]-(D_ref[kk]-ref_mean)*sci_mean/ref_mean-back_mean
    return D_sci, D_sub


############### evaluate this simulation ###############

def evaluation(nn, TT):
    global tra_one, z
    tra_one = np.load('Data/%s_%d.npy'%(pattern, T_star[TT]))
    z = np.load('Data/parameter_%d.npy'%T_star[TT])
    D_sci, D_sub = gain_calibration(nn, TT)
    for ll in range(len(lamb)):
        D_sci_bin = np.mean([[D_sci[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sci_bin_norm = D_sci_bin/np.mean([D_sci_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_sub_bin = np.mean([[D_sub[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sub_bin_norm = D_sub_bin/np.mean([D_sub_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_mod = tra_one[ll]
        D_sci_flat = average(TT, D_sci[ll])
        D_sub_flat = average(TT, D_sub[ll])
        D_mod_norm = D_mod/D_mod[10]
        D_sci_flat_norm = D_sci_flat/D_sci_flat[10]
        D_sub_flat_norm = D_sub_flat/D_sub_flat[10]
        sci_evaluation[nn, TT, ll] = (D_mod_norm[int(t_one[TT]/2)]-D_sci_flat_norm[int(t_one[TT]/2)])*10**6
        sub_evaluation[nn, TT, ll] = (D_mod_norm[int(t_one[TT]/2)]-D_sub_flat_norm[int(t_one[TT]/2)])*10**6
        sigma_top_sci = [D_sci_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]]
        sigma_bottom_sci = [D_sci_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]]
        sigma_top_sub = [D_sub_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]]
        sigma_bottom_sub = [D_sub_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]]
        sci_sigma[nn, TT, ll] = np.sqrt(np.std(sigma_top_sci)**2/len(sigma_top_sci)+np.std(sigma_bottom_sci)**2/len(sigma_bottom_sci))*10**6
        sub_sigma[nn, TT, ll] = np.sqrt(np.std(sigma_top_sub)**2/len(sigma_top_sub)+np.std(sigma_bottom_sub)**2/len(sigma_bottom_sub))*10**6


############### option ###############

def average(TT, X):
    if len(X)==t_one[TT]:
        average = X
    else:
        average = np.mean([[X[kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
    top = np.mean([average[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]])
    bottom = np.mean([average[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]])
    for kk in range(t_one[TT]):
        if z[kk]>1+p[TT]:
            average[kk] = top
        elif z[kk]<1-p[TT]:
            average[kk] = bottom
    return average


##############################

if __name__ == '__main__':
    main()

##############################
