# zodi.py
import numpy as np
import matplotlib.pyplot as plt

zodi_pattern = 0
t_exp = 60                      # exposure time(sec)
h = 6.626070040*10**(-27)       # plank constance(erg*sec)
c = 2.99792458*10**10           # light speed(cm/sec)
k = 1.38064852*10**(-16)        # boltzmann constance(erg/K)
T = 275                         # zodi temperature(K)
D = 9.24                        # telescope diameter(m)
A = 0.3                         # transmittance
dl = np.load('Data/dl.npy')     # wavelength band(µm)
lamb = np.load('Data/lamb.npy') # wavelength(µm)
zl = 5                          # zodi level at 9µm(MJy)
if zodi_pattern==0:
    ac = 1.5                    # field stop(arcsec)
elif zodi_pattern==1:
    ac = 1.0                    # field stop(arcsec)
elif zodi_pattern==2:
    ac = 2.0                    # field stop(arcsec)
else:
    ac = 4.0                    # field stop(arcsec)

zodi = np.array([0]*len(lamb), float)
zodi_9 = 8.70*10**(-12)*(9*10**(-4)/(h*c))*zl*t_exp*np.pi*(D/2)**2*A*np.pi*ac**2*dl[1]
for ll in range(len(lamb)):
    if lamb[ll]<6:
        dlamb = dl[0]
    elif 6<=lamb[ll]<11:
        dlamb = dl[1]
    else:
        dlamb = dl[2]
    zodi[ll] =  (2*h*c**2)/((lamb[ll]*10**(-4))**5*(np.exp((h*c)/(lamb[ll]*10**(-4)*k*T))-1))*dlamb/(h*c/(lamb[ll]*10**(-4)))
zodi *= zodi_9/zodi[np.where(abs(lamb-9)==np.min(abs(lamb-9)))]
plt.plot(lamb, zodi, 'ro', markersize=3)
plt.title('Zodiacal light')
plt.xlabel('Wavelength (µm)')
plt.ylabel('Flux (electron)')
plt.xlim(lamb[0], lamb[-1])
plt.show()
np.save('Data/zodi_{}.npy'.format(zodi_pattern), zodi)
