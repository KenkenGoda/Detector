# wavelength.py
import numpy as np

lamb_bord = np.array([3, 6, 11, 22])                # wavelength borders
R = np.array([100, 100, 100])                       # resolutions of each detector
lamb_sum = np.array([lamb_bord[0]+lamb_bord[1],
                     lamb_bord[1]+lamb_bord[2],
                     lamb_bord[2]+lamb_bord[3]])
dl = lamb_sum/2/R                                   # wavelength bands
lamb = []
lamb_start = lamb_bord[0]

while True:
    if lamb_start<lamb_bord[1]:
        lamb_mid = lamb_start+dl[0]/2
        lamb_start += dl[0]
        if lamb_start+dl[0]>lamb_bord[1]:
            lamb_start = lamb_bord[1]
    elif lamb_bord[1]<=lamb_start<lamb_bord[2]:
        lamb_mid = lamb_start+dl[1]/2
        lamb_start += dl[1]
        if lamb_start+dl[1]>lamb_bord[2]:
            lamb_start = lamb_bord[2]
    else:
        lamb_mid = lamb_start+dl[2]/2
        lamb_start += dl[2]
        if lamb_start+dl[2]>lamb_bord[3]:
            lamb.append(lamb_mid)
            break
    lamb.append(lamb_mid)

lamb = np.array(lamb)
print(lamb_bord)
print(dl)
print(lamb)
np.save('Data/lamb_bord.npy', lamb_bord)
np.save('Data/dl.npy', dl)
np.save('Data/lamb.npy', lamb)
