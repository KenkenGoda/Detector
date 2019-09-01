# transit_curve_matsuo.py
import numpy as np
import matplotlib.pylab as plt
import time

h = 6.626070040*10**(-27)                   # Plank constant(erg*s)
c = 2.99792458*10**10                       # speed of light(cm/s)
k = 1.38064852*10**(-16)                    # Boltzmann constant(erg/K)
pc = 3.085677581*10**18                     # 1 parsec(cm)
d = 10*pc                                   # distance(cm)
R_Sun = 6.960*10**10                        # Solar radius(cm)
R_Earth = 6.378137*10**8                    # Earth radius(cm)
R = np.load('Data/radius.npy')              # star radius(R_Sun)
R_star = R*R_Sun                            # star radius(cm)
R_planet = R_Earth                          # planet radius(cm)
p = R_planet/R_star                         # radius ratio
a = np.load('Data/semi.npy')                # semi major axis(cm)
i = 90*(np.pi/180)                          # inclination(rad)
P = np.load('Data/period.npy')              # period(sec)
o = 2*np.pi/P                               # orbital angular speed(rad/min)
T_star = np.load('Data/temperature.npy')    # star temperature(K)
T_planet = 288.2                            # planet temperature(K)
D = 924                                     # telescope diameter(cm)
t_exp = 12                                  # exposure time(sec)
A = 0.3                                     # transmittance
lamb_bord = np.load('Data/lamb_bord.npy')   # wavelength border(µm)
dl = np.load('Data/dl.npy')                 # wavelength band(µm)
lamb = np.load('Data/lamb.npy')             # wavelength(µm)
u = np.array([0]*len(lamb),float)           # limb darkening coefficient
P_tra = P*R_star/np.pi/a                    # transit period(sec)
t_one = P_tra*3/t_exp                       # the number of plots in one transit observation
t_one = t_one.astype(np.int64)
t = [np.linspace(1, t_one[TT], t_one[TT]) for TT in range(len(T_star))]    #time(min)


def main():
    for TT in range(len(T_star)):
        photon_number(TT)
        parameter(TT)
        primary(TT)
        secondary(TT)


def photon_number(TT):    # assume star and planet are blackbody
    global F_star, F_planet
    F_star = np.array([0]*len(lamb), float)
    F_planet = np.array([0]*len(lamb), float)
    for ll in range(len(lamb)):
        if lamb[ll]<lamb_bord[1]:
            dlamb = dl[0]
        elif lamb_bord[1]<=lamb[ll]<lamb_bord[2]:
            dlamb = dl[1]
        else:
            dlamb = dl[2]
        I_star = (2*h*c**2)/((lamb[ll]*10**(-4))**5*(np.exp((h*c)/(lamb[ll]*10**(-4)*k*T_star[TT]))-1)) # spectal radiance of star
        I_planet = (2*h*c**2)/((lamb[ll]*10**(-4))**5*(np.exp((h*c)/(lamb[ll]*10**(-4)*k*T_planet))-1)) # spectal radiance of planet
        E = (h*c)/(lamb[ll]*10**(-4))                                                                   # energy of photon
        F_star[ll] = (I_star/E)*t_exp*(np.pi*(D/2)**2)*(1/d**2)*(dlamb*10**(-4))*A                      # flux of star
        F_planet[ll] = (I_planet/E)*t_exp*(np.pi*(D/2)**2)*(1/d**2)*(dlamb*10**(-4))*A                  # flux of planet


def parameter(TT):
    global z, a0, a1, a2
    z = np.array([0]*t_one[TT], float)
    a0 = np.array([0]*t_one[TT], float)
    a1 = np.array([0]*t_one[TT], float)
    a2 = np.array([0]*t_one[TT], float)
    for kk in range(t_one[TT]):
        s = a[TT]*np.sqrt((np.sin(o[TT]*(kk-(t_one[TT]/2-0.5))*t_exp))**2+((np.cos(i))*(np.cos(o[TT]*(kk-(t_one[TT]/2-0.5))*t_exp)))**2)    #planet position from central star
        z[kk] = s/R_star[TT]
        a0[kk] = np.arcsin((p[TT]/z[kk]))
        a1[kk] = np.arccos((z[kk]**2+p[TT]**2-1)/(2*p[TT]*z[kk]))
        a2[kk] = np.arccos((z[kk]**2-p[TT]**2+1)/(2*z[kk]))
    np.save('Data/parameter_matsuo_%d.npy'%T_star[TT], z)


def primary(TT):      # primary transmission
    start = time.time()
    pri_curve = np.array([[0]*t_one[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        for kk in range(t_one[TT]):
            N = 10**4
            if z[kk]>=1+p[TT]:
                S = 0
            elif 1-p[TT]<= z[kk]<1+p[TT]:
                S = R_star[TT]**2*(p[TT]**2*a1[kk]+a2[kk]-(np.sqrt(4*z[kk]**2-(z[kk]**2-p[TT]**2+1)**2))/2)
            else:
                S = np.pi*R_planet**2
            pri_curve[ll, kk] = (np.pi*R_star[TT]**2-S)*F_star[ll]+np.pi*R_planet**2*F_planet[ll]
        if TT==0 and ll==0:
            plt.plot(t[TT], pri_curve[ll])
            plt.xlim(0, t_one[TT])
            plt.show()
        elapsed_time = time.time()-start
        print('%f: %dh%dm%ds'%(lamb[ll], int(elapsed_time/3600), int((elapsed_time-int(elapsed_time/3600)*3600)/60), elapsed_time-int(elapsed_time/60)*60))
    np.save("Data/primary_matsuo_%d.npy"%T_star[TT], pri_curve)


def secondary(TT):        # secondary transmission
    sec_curve = np.array([[0]*t_one[TT]]*len(lamb), float)
    for ll in range(len(lamb)):
        F_full = F_star[ll]*(3-u[ll])/3
        for kk in range(t_one[TT]):
            if z[kk] > 1+p[TT]:
                S = 0                                                                                       # eclipsed area
            elif 1-p[TT]<z[kk]<=1+p[TT]:
                S = R_star[TT]**2*(p[TT]**2*a1[kk]+a2[kk]-(np.sqrt(4*z[kk]**2-(z[kk]**2-p[TT]**2+1)**2))/2) # eclipsed area
            else:
                S = np.pi*R_planet**2                                                                       # eclipsed area
            sec_curve[ll,kk] = np.pi*R_star[TT]**2*F_full+(np.pi*R_planet**2-S)*F_planet[ll]
    np.save("Data/secondary_matsuo_%d.npy"%T_star[TT], sec_curve)


if __name__ == ('__main__'):
    main()
