# discussion_flat.py
import numpy as np
from numpy import random as rd
from scipy.stats import norm
import pyfits
import matplotlib.pyplot as plt
import time

start = time.time()

############### changable parameter ###############

hor_pix = 300                               # horizontal size of science and background pixel(pix)
ver_pix = 900                               # vertical size of science and background pixel(pix)
n = 100                                     # repeat times
T_star_len = 4
t_exp = 60                                  # exposure time(sec)
times = 60                                  # number of transmissions
Q = 1.0*t_exp                               # dark current value(electron)
gain_std = 100                              # standard deviation of gain(ppm)
pattern = 'primary'                         # transit pattern ('primary' or 'secondary')
wog = False                                 # if the signal has gain (wog: without-gain)
lamb_bin = True                             # if the backgroud signal is binned along wavelength


############### constant parameter ###############

h = 6.626070040*10**(-27)                   # plank constant(erg*s)
c = 2.99792458*10**10                       # speed of light(cm/s)
k = 1.38064852*10**(-16)                    # boltzmann constant(erg/K)
T_star = np.load('Data/temperature.npy')    # star temperature(K)
T_planet = 288.2                            # planet temperature(K)
R_Sun = 6.960*10**10                        # solar radius(cm)
R_Earth = 6.378137*10**8                    # earth radius(cm)
R = np.load('Data/radius.npy')              # star radius(R_Sun)
R_star = R*R_Sun                            # star radius(cm)
R_planet = R_Earth                          # planet radius(cm)
p = R_planet/R_star                         # radius ratio
a = np.load('Data/semi.npy')                # semi major axis(cm)
P = np.load('Data/period.npy')              # period(sec)
P_tra = P*R_star/np.pi/a                    # transit period(sec)
t_one = P_tra*3/t_exp                       # the number of plots in one transit observation
t_one = t_one.astype(np.int64)
t_total = t_one*times                       # total times(2min)
t0 = [np.linspace(1, t_one[TT], t_one[TT]) for TT in range(len(T_star))]    # time for one transmission(min)
t = [np.linspace(1, t_total[TT], t_total[TT]) for TT in range(len(T_star))] # time for total(min)
lamb_bord = np.load('Data/lamb_bord.npy')   # wavelength borders(µm)
dl = np.load('Data/dl.npy')                 # wavelength bands(µm)
lamb = np.load('Data/lamb.npy')             # wavelength(µm)
ver_1 = int(ver_pix*dl[0]/(lamb_bord[1]-lamb_bord[0]))  # the number of vertical science pixels by wavelength on detector1
ver_2 = int(ver_pix*dl[1]/(lamb_bord[2]-lamb_bord[1]))  # the number of vertical science pixels by wavelength on detector2
ver_3 = int(ver_pix*dl[2]/(lamb_bord[3]-lamb_bord[2]))  # the number of vertical science pixels by wavelength on detector3
sci_pix_1 = hor_pix*ver_1               # the number of science pixels by wavelength on detector1
sci_pix_2 = hor_pix*ver_2               # the number of science pixels by wavelength on detector2
sci_pix_3 = hor_pix*ver_3               # the number of science pixels by wavelength on detector3
back_pix_1 = sci_pix_1                      # the number of background pixels by wavelength on detector1
back_pix_2 = sci_pix_2                      # the number of background pixels by wavelength on detector2
back_pix_3 = sci_pix_3                      # the number of background pixels by wavelength on detector3
ref_pix = 1024*1024-200*600*2               # the number of reference pixels on detector1, 2 and 3
fowler = 16                                 # number of fowler sampling
read = 21.9/np.sqrt(fowler)                 # readout noise(electron/read)
n_gate = 4                                  # the number of gates
zodi = np.load('Data/zodi.npy')             # zodiacal light by wavelength(electron)


############### main ###############

def main():
    global sub_evaluation_1, sub_sigma_1
    global sub_evaluation_2, sub_sigma_2
    global sub_evaluation_3, sub_sigma_3
    global flat
    print('hor_pix: {}'.format(hor_pix))
    print('ver_pix: {}'.format(ver_pix))
    print('repeat: {}'.format(n))
    print('pattern: {}'.format(pattern))
    print('wog: {}'.format(wog))
    print('lamb_bin: {}'.format(lamb_bin))
    sub_evaluation_1 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    sub_evaluation_2 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    sub_evaluation_3 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    sub_sigma_1 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    sub_sigma_2 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    sub_sigma_3 = np.array([[[0]*len(lamb)]*T_star_len]*n, float)
    flat = flat_field()

    plt.scatter(lamb, flat)
    plt.xlabel('Wavelength [µm]')
    plt.ylabel('Residual')
    plt.xlim(lamb[0]-0.1, lamb[-1]+0.1)
    plt.ylim(flat.min()-(flat.max()-flat.min())*0.1, flat.max()+(flat.max()-flat.min())*0.1)
    plt.show()
    
    [evaluation(nn, TT) for nn in range(n) for TT in range(T_star_len)]
    np.save('Result/sub_evaluation_both_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_evaluation_1)
    np.save('Result/sub_evaluation_back_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_evaluation_2)
    np.save('Result/sub_evaluation_ref_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_evaluation_3)
    np.save('Result/sub_sigma_both_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_sigma_1)
    np.save('Result/sub_sigma_back_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_sigma_2)
    np.save('Result/sub_sigma_ref_flat_{}.npy'.format(int(hor_pix*ver_pix)), sub_sigma_3)
    print('Success')


############### make the flat field ###############

def flat_field():
    res_1 = residual('short')
    res_2 = residual('middle')
    res_3 = residual('long')
    return np.hstack((res_1, res_2, res_3))

def residual(case):
    if case=='short':
        lamb_len = len(lamb[lamb<6])
        ver = ver_1
    elif case=='middle':
        lamb_len = len(lamb[(lamb>=6)&(lamb<11)])
        ver = ver_2
    else:
        lamb_len = len(lamb[lamb>=11])
        ver = ver_3
    sci_start = [int((1024-ver_pix)/2), int((1024-(400+hor_pix))/2)]
    back_start = [sci_start[0], sci_start[1]+400]
    sci = np.array([pixel_averaged(ll, sci_start, ver) for ll in range(lamb_len)])
    back = np.array([pixel_averaged(ll, back_start, ver) for ll in range(lamb_len)])
    return sci-back

def pixel_averaged(ll, pix_start, ver):
    flat_unc = pyfits.open('Data/wise-w3-flat_unc.fits')[0].data
    ver_start = pix_start[0]+ver*ll
    ver_end = ver_start+ver
    hor_start = pix_start[1]
    hor_end = hor_start+hor_pix
    flat_unc_tar = flat_unc[ver_start:ver_end, hor_start:hor_end]
    return np.mean(flat_unc_tar[flat_unc_tar<=0.002])


############### make the gain fluctuation ###############

def gain_fluctuation(TT):
    period = np.linspace(60*2, t_one[TT]*60*4, 10000)
    frequency = 1/period
    fluctuation = np.sum([Fluctuation(f, t0[TT]*60) for f in frequency], axis=0)*np.sqrt(frequency[-1])
    fluctuation = fluctuation*(gain_std*10**(-6)/fluctuation.std())
    gain_one = rd.normal(1, 0.00015)+fluctuation
    gain = gain_one
    for _ in range(times-1):
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
    D_sci_ijk = np.array([rd.normal((tra[kk]+zodi[ll])/n_pix+Q,
                                   np.sqrt(((bose_star*tra[kk]+bose_zodi*zodi[ll])/n_pix+Q+read)/(n_pix/n_gate)))
                          for kk in range(t_total[TT])], float)*gain
    return D_sci_ijk


############### make the background curves ###############

def background(TT, ll, gain, n_pix):
    x_zodi = (h*c)/(lamb[ll]*10**(-4)*k*275)
    bose_zodi = abs(np.exp(x_zodi)/(np.exp(x_zodi)-1))
    D_back_ijk = norm.rvs(zodi[ll]/n_pix+Q,
                          np.sqrt(((bose_zodi*zodi[ll])/n_pix+Q+read)/(n_pix/n_gate)), t_total[TT])*gain
    return D_back_ijk


############### make the reference curves ###############

def reference(TT, gain, n_pix):
    D_ref_ijk = norm.rvs(Q, np.sqrt((Q+read)/(n_pix/n_gate)), t_total[TT])*gain
    return D_ref_ijk


############### each average curve ###############

def unify(nn, TT):
    D_sci_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_back_ave_k = np.array([[[0]*t_total[TT]]*len(lamb)]*n_gate, float)
    D_ref_ave_k = np.array([[[0]*t_total[TT]]*n_gate]*3, float)
    for jj in range(n_gate):
        if wog==True:
            gain = 1.0
        else:
            gain_1 = gain_fluctuation(TT)
            gain_2 = gain_fluctuation(TT)
            gain_3 = gain_fluctuation(TT)
        for ll in range(len(lamb)):
            if lamb[ll]<lamb_bord[1]:
                sci_pix = sci_pix_1
                back_pix = back_pix_1
                if wog==False:
                    gain = gain_1
            elif lamb_bord[1]<=lamb[ll]<lamb_bord[2]:
                sci_pix = sci_pix_2
                back_pix = back_pix_2
                if wog==False:
                    gain = gain_2
            else:
                sci_pix = sci_pix_3
                back_pix = back_pix_3
                if wog==False:
                    gain = gain_3
            D_sci_ave_k[jj, ll] = science(TT, ll, gain, sci_pix)+flat[ll]
            D_back_ave_k[jj, ll] = background(TT, ll, gain, back_pix)+flat[ll]
        if wog==True:
            D_ref_ave_k[0, jj] = reference(TT, gain, ref_pix)+flat[ll]
            D_ref_ave_k[1, jj] = reference(TT, gain, ref_pix)+flat[ll]
            D_ref_ave_k[2, jj] = reference(TT, gain, ref_pix)+flat[ll]
        else:
            D_ref_ave_k[0, jj] = reference(TT, gain_1, ref_pix)+flat[ll]
            D_ref_ave_k[1, jj] = reference(TT, gain_2, ref_pix)+flat[ll]
            D_ref_ave_k[2, jj] = reference(TT, gain_3, ref_pix)+flat[ll]
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
    D_back_all_1 = np.mean(D_back_ave[np.where(lamb<lamb_bord[1])], axis=0)
    D_back_all_2 = np.mean(D_back_ave[np.where((lamb>=lamb_bord[1])&(lamb<lamb_bord[2]))], axis=0)
    D_back_all_3 = np.mean(D_back_ave[np.where(lamb>=lamb_bord[2])], axis=0)
    D_sub_1 = np.array([[0]*t_total[TT]]*len(lamb), float)
    D_sub_2 = np.array([[0]*t_total[TT]]*len(lamb), float)
    D_sub_3 = np.array([[0]*t_total[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        if lamb[ll]<lamb_bord[1]:
            D_back_all = D_back_all_1
            D_ref = D_ref_ave[0]
        elif lamb_bord[1]<=lamb[ll]<lamb_bord[2]:
            D_back_all = D_back_all_2
            D_ref = D_ref_ave[1]
        else:
            D_back_all = D_back_all_3
            D_ref = D_ref_ave[2]
        for tt in range(times):
            sci_mean = 0
            back_mean = 0
            back_all_mean = 0
            ref_mean = 0
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                sci_mean += D_sci_ave[ll, kk]
                back_mean += D_back_ave[ll, kk]
                back_all_mean += D_back_all[kk]
                ref_mean += D_ref[kk]
            sci_mean /= t_one[TT]
            back_mean /= t_one[TT]
            back_all_mean /= t_one[TT]
            ref_mean /= t_one[TT]
            for kk in range(tt*t_one[TT], (tt+1)*t_one[TT]):
                D_sci[ll, kk] = D_sci_ave[ll, kk]-back_mean
                if lamb_bin==False:
                    D_sub_1[ll, kk] = D_sci_ave[ll, kk]-((D_back_ave[ll, kk]+D_ref[kk])-(back_mean+ref_mean))*sci_mean/(back_mean+ref_mean)-back_mean
                    D_sub_2[ll, kk] = D_sci_ave[ll, kk]-(D_back_ave[ll, kk]-back_mean)*sci_mean/back_mean-back_mean
                else:
                    D_sub_1[ll, kk] = D_sci_ave[ll, kk]-((D_back_all[kk]+D_ref[kk])-(back_all_mean+ref_mean))*sci_mean/(back_all_mean+ref_mean)-back_mean
                    D_sub_2[ll, kk] = D_sci_ave[ll, kk]-(D_back_all[kk]-back_all_mean)*sci_mean/back_all_mean-back_mean
                D_sub_3[ll, kk] = D_sci_ave[ll, kk]-(D_ref[kk]-ref_mean)*sci_mean/ref_mean-back_mean
    return D_sci, D_sub_1, D_sub_2, D_sub_3


############### evaluate this simulation ###############

def evaluation(nn, TT):
    global tra_one, z
    tra_one = np.load('Data/%s_%d.npy'%(pattern, T_star[TT]))
    z = np.load('Data/parameter_%d.npy'%T_star[TT])
    D_sci, D_sub_1, D_sub_2, D_sub_3 = gain_calibration(nn, TT)
    for ll in range(len(lamb)):
        D_sci_bin = np.mean([[D_sci[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sci_bin_norm = D_sci_bin/np.mean([D_sci_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_sub_1_bin = np.mean([[D_sub_1[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sub_1_bin_norm = D_sub_1_bin/np.mean([D_sub_1_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_sub_2_bin = np.mean([[D_sub_2[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sub_2_bin_norm = D_sub_2_bin/np.mean([D_sub_2_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_sub_3_bin = np.mean([[D_sub_3[ll, kk+tt*t_one[TT]] for kk in range(t_one[TT])] for tt in range(times)], axis=0)
        D_sub_3_bin_norm = D_sub_3_bin/np.mean([D_sub_3_bin[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]], axis=0)
        D_mod = tra_one[ll]
        D_sub_1_flat = average(TT, D_sub_1[ll])
        D_sub_2_flat = average(TT, D_sub_2[ll])
        D_sub_3_flat = average(TT, D_sub_3[ll])
        D_mod_norm = D_mod/D_mod[10]
        D_sub_1_flat_norm = D_sub_1_flat/D_sub_1_flat[10]
        D_sub_2_flat_norm = D_sub_2_flat/D_sub_2_flat[10]
        D_sub_3_flat_norm = D_sub_3_flat/D_sub_3_flat[10]
        sub_evaluation_1[nn, TT, ll] = (D_mod_norm[int(t_one[TT]/2)]-D_sub_1_flat_norm[int(t_one[TT]/2)])*10**6
        sub_evaluation_2[nn, TT, ll] = (D_mod_norm[int(t_one[TT]/2)]-D_sub_2_flat_norm[int(t_one[TT]/2)])*10**6
        sub_evaluation_3[nn, TT, ll] = (D_mod_norm[int(t_one[TT]/2)]-D_sub_3_flat_norm[int(t_one[TT]/2)])*10**6
        sigma_top_sub_1 = [D_sub_1_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]]
        sigma_bottom_sub_1 = [D_sub_1_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]]
        sigma_top_sub_2 = [D_sub_2_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]]
        sigma_bottom_sub_2 = [D_sub_2_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]]
        sigma_top_sub_3 = [D_sub_3_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]>1+p[TT]]
        sigma_bottom_sub_3 = [D_sub_3_bin_norm[kk] for kk in range(t_one[TT]) if z[kk]<1-p[TT]]
        sub_sigma_1[nn, TT, ll] = np.sqrt(np.std(sigma_top_sub_1)**2/len(sigma_top_sub_1)+np.std(sigma_bottom_sub_1)**2/len(sigma_bottom_sub_1))*10**6
        sub_sigma_2[nn, TT, ll] = np.sqrt(np.std(sigma_top_sub_2)**2/len(sigma_top_sub_2)+np.std(sigma_bottom_sub_2)**2/len(sigma_bottom_sub_2))*10**6
        sub_sigma_3[nn, TT, ll] = np.sqrt(np.std(sigma_top_sub_3)**2/len(sigma_top_sub_3)+np.std(sigma_bottom_sub_3)**2/len(sigma_bottom_sub_3))*10**6

'''
        if nn==0 and TT==0 and ll==0:
            plt.plot(D_sci_bin_norm, label='sci')
            plt.plot(D_sub_1_bin_norm, label='both')
            plt.plot(D_sub_2_bin_norm, label='back')
            plt.plot(D_sub_3_bin_norm, label='ref')
            plt.plot(D_mod_norm, label='mod')
            plt.legend()
            plt.show()
'''


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
