# Detector.py
import numpy as np
import numpy.random as rd
from scipy.stats import norm
import matplotlib.pyplot as plt

au = 1.498978707e13     # 1 au (cm)
ps = 3.085677581e18     # 1 parsec (cm)
h = 6.626070040e-27     # Plank constant (erg*s)
c = 2.99792458e10       # the speed of light (cm/s)
k = 1.38064852e-16      # Boltzmann constant (erg/K)
G = 6.67408e-5          # gravitation constant (cm^3/kg/s^2)
R_Sun = 6.960e10        # Solar radius (cm)
R_Earth = 6.378137e8    # Earth radius (cm)
M_Sun = 1.989e30        # ths Sun mass (kg)


def main():
    T, R, a, M, P = Characterization.extract(Ts=Ts, Tp=Tp, alb=alb, acc=acc)
    lamb, dl = Wavelength.extract(lamb_min=lamb_min, lamb_max=lamb_max, R=R)
    Simulation()


class Characterization:

    def extract(Ts=None, Tp=288.2, alb=0.306, acc=8e-2/7):
        R = -8.133+5.09342e-3*Ts-9.86602e-7*Ts**2+6.47963e-11*Ts**3
        if Ts==2500:
            R = 0.1
        elif Ts==3000:
            R = 0.1+acc*5
        a = R*R_Sun/2*(Ts/Tp)**2*np.sqrt(1-alb)
        M = (-0.6063+np.sqrt(0.6063**2-4*0.3200*(0.0906-R)))/0.64*M_Sun
        P = 2*np.pi*np.sqrt(a**3/G/M)
        return Ts, R, a, M, P


class Wavelength:

    def extract(lamb_min=3, lamb_max=6, R=100):
        dl = (lamb_min+lamb_max)*0.5/R
        lamb = []
        center = lamb_min+dl*0.5
        while True:
            lamb.append(center)
            center += dl
            if center+dl*0.5>lamb_max:
                break
        return np.array(lamb), dl


class BlackBody:

    def extract(lamb=None, dl=dl, T=None, D=924, d=10, A=30, t_exp=60):
        lamb *= 10**(-4)
        S = np.pi*(D*0.5)**2
        d *= ps
        dl *= 10**(-4)
        A *= 0.01
        I = 2*np.pi*h*c**2/(lamb**5*(np.exp((h*c)/(lamb*k*T))-1))
        F = I*t_exp*S/d**2*dl*A
        E = h*c/lamb
        return F/E


class TransitCurve:

    def __init__(self, lamb=None, dl=None, Ts=None, Tp=288.2, Rs=None, Rp=R_Earth, a=None, P=None, i=90, alb=0.306, D=924, d=10, A=30, t_exp=60):
        self._lamb= lamb
        self._Rs = Rs
        self._Rp = Rp
        self._p = Rp/Rs
        self._o = 2*np.pi/P
        self._i = i*np.pi/180
        self._t_exp = t_exp
        self._t_one = int(P*Rs/np.pi/a*3/t_exp)
        self.Fs = BlackBody.extract(lamb=lamb, dl=dl, T=Ts, D=D, d=d, A=A, t_exp=t_exp)
        self.Fp = BlackBody.extract(lamb=lamb, dl=dl, T=Tp, D=D, d=d, A=A, t_exp=t_exp)

    def _parameter(self):
        sin = np.array([np.sin(self._o*(kk-(self._t_one*0.5-0.5))*self._t_exp) for kk in range(self._t_one)])
        cos = np.array([np.cos(self._i)*np.cos(self._o*(kk-(self._t_one*0.5-0.5))*self._t_exp) for kk in range(self._t_one)], float)
        s = self._a*np.sqrt(sin**2+cos**2)
        z = s/self._Rs
        a0 = np.arcsin(self._p/z)
        a1 = np.arccos((z**2+self._p**2-1)/(2*self._p*z))
        a2 = np.arccos((z**2-self._p**2+1)/(2*z))
        return z, a0, a1, a2

    def extract(self, case='primary'):
        z, a0, a1, a2 = self._parameter()
        tra = np.array([[0]*self._t_one]*len(self._lamb), float)
        for ll in range(len(self._lamb)):
            for kk in range(self._t_one):
                if z[kk]>=1+self._p:
                    S = 0
                elif 1-self._p<=z[kk]<1+self._p:
                    if case=='primary':
                        S = self._Rs**2*(self._p**2*a1[kk]+a2[kk]-np.sqrt(4*z[kk]**2-(z[kk]**2-self._p**2+1)**2)*0.5)
                    else:
                        S = self._Rs**2*(self._p**2*a1[kk]+a2[kk]-np.sqrt(4*z[kk]**2-(z[kk]**2-self._p**2+1)**2)*0.5)
                else:
                    S = np.pi*self._Rp**2
                if case=='primary':
                    tra[ll, kk] = (np.pi*self._Rs**2-S)*self._Fs[ll]+np.pi*self._Rp**2*self._Fp
                else:
                    tra[ll, kk] = np.pi*self._Rs**2*self._Fs+(np.pi*self._Rp**2-S)*self._Fp
        return tra


class ZODI:

    def extract(lamb=None, dl=None, T=275, D=924, field=2.0, A=30, level=5.0, t_exp=60):
        Sm = np.pi*(D*0.01*0.5)**2
        ac2 = np.pi*field**2
        dl *= 10**(-4)
        zodi = BlackBody.extract(lamb=lamb, dl=dl, T=T)
        zodi_9 = 8.70e-12*level*t_exp*Sm*ac2*A*dl*(9e-4/(h*c))
        zodi *= zodi_9/zodi[np.where(abs(lamb-9)==np.min(abs(lamb-9)))]
        return zodi


class Simulation:

    def __init__(self, case='primary', lamb=None, dl=None, Ts=None, Tp=288.2, Tz=275, Rs=None, Rp=R_Earth, a=None, P=None, i=90, alb=0.306, D=924, d=10, field=2.0, A=30, level=5.0, t_exp=60, times=60, xpix=1024, ypix=1024, sci_xpix=200, sci_ypix=600, n_gate=4, dark=1.0, fowler=16, gain_std=100):
        self._lamb = lamb
        self._Ts = Ts
        self._Tz = Tz
        self._t_exp = t_exp
        self._times = times
        self._t_one = int(P*Rs/np.pi/a*3/t_exp)
        self._t_total = self._t_one*times
        self._t0 = np.arange(1, self._t_one+1)
        self._t = np.arange(1, self._t_total+1)
        self._sci_pix = sci_xpix*int(sci_ypix*dl/(lamb[-1]-lamb[0]))
        self._back_pix = self._sci_pix
        self._ref_pix = xpix*ypix-sci_xpix*sci_ypix*2
        self._gate = n_gate
        self._Q = dark*t_exp
        self._read = 21.9/np.sqrt(fowler)
        self._std = gain_std
        self._tra = TransitCurve(lamb=lamb, dl=dl, Ts=Ts, Tp=Tp, Rs=Rs, Rp=Rp, a=a, P=P, i=i, alb=alb, D=D, d=d, A=A, t_exp=t_exp).extract(case=case)
        self._zodi = ZODI.extract(lamb=lamb, dl=dl, T=Tz, D=D, field=field, A=A, level=level, t_exp=t_exp)

    def _gain_fluctuation(self):
        period = np.linspace(self._t_exp*2, self._t_one*self._t_exp*4, 10000)
        frequency = 1/period
        fluctuation = np.sum([np.sin(2*np.pi*(ff*self._t0*self._t_exp+rd.random()))/np.sqrt(ff) for ff in frequency], axis=0)
        fluctuation = fluctuation*(self._std*10**(-6)/fluctuation.std())
        gain_one = rd.normal(1, 0.00015)+fluctuation
        gain = np.concatenate([gain_one for _ in range(self._times)])
        return gain

    def _science(self, ll, gain):
        tra = np.concatenate([self._tra[ll] for _ in range(self._times)])
        Xs = (h*c)/(self._lamb*10**(-4)*k*self._Ts)
        Xz = (h*c)/(self._lamb*10**(-4)*k*self._Tz)
        Bs = abs(np.exp(Xs)/(np.exp(Xs)-1))
        Bz = abs(np.exp(Xz)/(np.exp(Xz)-1))
        Dsci = np.array([rd.normal((tra[kk]+self._zodi[ll])/self._sci_pix+self._Q, np.sqrt(((Bs*tra[kk]+Bz*self._zodi[ll])/self._sci_pix+self._Q+self._read)/(self._sci_pix/self._gate))) for kk in range(self._t_total)], float)*gain
        return Dsci
        
    def _background(self, ll, gain):
        Xz = (h*c)/(self._lamb*10**(-4)*k*self._Tz)
        Bz = abs(np.exp(Xz)/(np.exp(Xz)-1))
        Dback = norm.rvs(self._zodi[ll]/self._back_pix+self._Q, np.sqrt(((Bz*self._zodi[ll])/self._back_pix+self._Q+self._read)/(self._back_pix/self._gate)), self._t_total)*gain
        return Dback

    def _reference(self, gain):
        Dref = norm.rvs(self._Q, np.sqrt((self._Q+self._read)/(self._ref_pix/self._gate)), self._t_total)*gain
        return Dref

    def _unify(self):



if __name__==('__main__'):
    main()
