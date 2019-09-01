#!/usr/bin/env python



#***************************************************************************************
# This is a python version ot the program transit_spec_noise_file.py
# It has been modified to create a stellar spectrum using pysynphot (Phoenix models) and to plot of the star / planet spectrum
#
# TRANSIT_SPEC computes the transmission spectrum of an exoplanet
# passing in front of a model stellar spectrum
# It takes the files of a stellar spectrum
# and a file of exoplanet radius with wavelength.
#
#
# This version also reads a file of limiting  fractional precision at
# different wavelengths. The outputs are:
#  wavelength, fluxes or ratios with noise, noiseless values, and 1 sigma noise
#
#
#***************************************************************************************

# basic python packages
import numpy as np
import matplotlib.pyplot as plt
import astropy # basic astrotools
from astropy import units as u
import pysynphot as S

from math import *
from sys import *
from time import *
from random import *
import operator


# global variables
#MAXPOINTS = 50000
PI = np.pi

# telescope properties
OST9AREA  = 650000  #  Unobscured 9.1-m diameter OST has area 65.0 m^2
OST6AREA  = 250000  #  Obscured 5.8-m diameter OST
OST5AREA  = 196000  #  Unobscured 5-m diameter OST has area 19.6 m^2
NPUPIL    = 5       #  Number of OST pupil slices
JWSTAREA = 250000  # JWST area in sq cm
HSTAREA = 45000  # HST primary area in sq cm
FINESSEAREA = 4536 # FINESSE primary area in sq cm

#Detector noise and dark current
CDS_HgCdTe = 20  # electrons CDS
CDS_SiAs = 35.4  # electrons CDS = 25e- (per read; M Ressler Feb 2, 2017) * sqrt(2) : previously used CDS = 28
CDS_InSb = 20  # Estimate given IRAC noise = 8.3e- Fowler-32
DC_HgCdTe_5um  = 0.01
DC_HgCdTe_10um = 1.0
DC_SiAs = 0.1

# physical constants
h = 6.626e-27  # cgs units
c = 2.998e10
k = 1.381e-16


# Star model file of wavelength (microns) and Flux (cgs)
class Star:
    def __init__(self, wav, F):
        self.wav = wav  # a float
        self.F = F  # a float
    def __repr__(self):
        return "Star(%r, %r)" %(self.wav, self.F)
    def __eq__(self, other):
        return type(other) == Star and self.wav == other.wav and self.F == other.F

# Planet model file of wavelength (microns) and radius (km)
class Planet:
    def __init__(self, wav, rad):
        self.wav = wav  # a float
        self.rad = rad  # a float
    def __repr__(self):
        return "Planet(%r, %r)" %(self.wav, self.rad)
    def __eq__(self, other):
        return type(other) == Planet and self.wav == other.wav and self.rad == other.rad

"""
Systematic noise file of wavelength (microns) and max SNR  (1 / precision)
NOTE: This is the precision for individual measurements of the star and the [(star + pl) - star]
I assume this is ALSO the precision of [(star + pl) - star] / star ; not sqrt worse
"""
class Noise:
    def __init__(self, wav, pre):
        self.wav = wav  # a float
        self.pre = pre  # a float
    def __repr__(self):
        return "Noise(%r, %r)" %(self.wav, self.pre)
    def __eq__(self, other):
        return type(other) == Noise and self.wav == other.wav and self.pre == other.pre




#------------------------------------------------------------------------------------------------------
# Functions
#------------------------------------------------------------------------------------------------------

# compares two strings
def strcmp(str1,str2):
    if (str1 == str2):
        return 0
    if (str1 > str2) or (str2 > str1):
        return -1

def elec(lam, R, flux, t, mode):
    tel_area = 0.0  # set tel area to 0; set telescope in each mode

    if (strcmp(mode,"gr700xd") == 0):  # NIRISS X-disp R=700 grism mode
        tel_area = JWSTAREA  # using JWST telescope
        eff = 0
        """
        gr700xd grism throughputs are from: jwst_niriss_gr700xd-ord1_speceff.fits,
        file ~/WD/NGST/Instruments/NIRISS/jwst_niriss_gr700xd-order_m_1_spec_eff.dat
        Telescope reflectivity from ~/WD/NGST/JWST_telescope_OTE_throughput_sept2015.txt
        and assume NIRISS optics = 0.6
        """
        eff = 0.01
        if (lam > 0.65): eff = 0.017 * 0.66 * 0.6;
        if (lam > 0.70): eff = 0.09 * 0.75 * 0.6;
        if (lam > 0.75): eff = 0.18 * 0.78 * 0.6;
        if (lam > 0.80): eff = 0.30 * 0.81 * 0.6;
        if (lam > 0.85): eff = 0.41 * 0.82 * 0.6;
        if (lam > 0.90): eff = 0.53 * 0.84 * 0.6;
        if (lam > 0.95): eff = 0.62 * 0.85 * 0.6;
        if (lam > 1.00): eff = 0.70 * 0.85 * 0.6;
        if (lam > 1.05): eff = 0.76 * 0.86 * 0.6;
        if (lam > 1.10): eff = 0.81 * 0.86 * 0.6;
        if (lam > 1.15): eff = 0.82 * 0.86 * 0.6;
        if (lam > 1.20): eff = 0.86 * 0.86 * 0.6;
        if (lam > 1.30): eff = 0.86 * 0.86 * 0.6;
        if (lam > 1.40): eff = 0.82 * 0.86 * 0.6;
        if (lam > 1.50): eff = 0.77 * 0.87 * 0.6;
        if (lam > 1.60): eff = 0.71 * 0.87 * 0.6;
        if (lam > 1.70): eff = 0.65 * 0.87 * 0.6;
        if (lam > 1.80): eff = 0.59 * 0.87 * 0.6;
        if (lam > 1.90): eff = 0.53 * 0.88 * 0.6;
        if (lam > 2.00): eff = 0.49 * 0.88 * 0.6;
        if (lam > 2.10): eff = 0.44 * 0.88 * 0.6;
        if (lam > 2.20): eff = 0.40 * 0.88 * 0.6;
        if (lam > 2.30): eff = 0.37 * 0.88 * 0.6;
        if (lam > 2.40): eff = 0.34 * 0.88 * 0.6;
        if (lam > 2.50): eff = 0.31 * 0.89 * 0.6;
        if (lam > 2.60): eff = 0.28 * 0.89 * 0.6;
        if (lam > 2.70): eff = 0.26 * 0.89 * 0.6;
        if (lam > 2.80): eff = 0.23 * 0.88 * 0.6;

    if (strcmp(mode, "ncgsw") == 0):  # NIRCam SW grism mode
        tel_area = JWSTAREA  # using JWST telescope */
        eff = 0
        if (lam < 1.5):
            eff = 0.1 + (lam - 1.0) * 1.2
        else:
            eff = 0.7 + (1.5 - lam)
        eff = eff * 0.9 * 0.7 * 0.85 * 0.9  # tele, optics, filter, detector

    if (strcmp(mode, "ncglw") == 0):  # NIRCam LW grism mode
        tel_area = JWSTAREA  # using JWST telescope
        # throughputs are from  NIRCam_throughputs_BHilbert_2016Feb (STScI)
        # Values with grism are computed / taken from NIRCam_grism_blaze_function.xls
        eff = 0.08  # grism_eff * tel * optics+F322W2 * QE : 2.46 microns
        if (lam > 2.45): eff = 0.105;  # Values from NIRCam_grism_blaze_function.xls May 2016 STScI
        if (lam > 2.50): eff = 0.13;
        if (lam > 2.60): eff = 0.16;
        if (lam > 2.70): eff = 0.19;
        if (lam > 2.80): eff = 0.22;
        if (lam > 2.90): eff = 0.23;
        if (lam > 3.00): eff = 0.23;
        if (lam > 3.10): eff = 0.23;
        if (lam > 3.20): eff = 0.27;
        if (lam > 3.30): eff = 0.31;
        if (lam > 3.35): eff = 0.29;
        if (lam > 3.40): eff = 0.32;
        if (lam > 3.50): eff = 0.34;
        if (lam > 3.60): eff = 0.35;
        if (lam > 3.70): eff = 0.36;
        if (lam > 3.80): eff = 0.36;
        if (lam > 3.90): eff = 0.35;  # F444W starts here
        if (lam > 4.00): eff = 0.34;
        if (lam > 4.10): eff = 0.34;
        if (lam > 4.20): eff = 0.33;
        if (lam > 4.30): eff = 0.32;
        if (lam > 4.40): eff = 0.31;
        if (lam > 4.50): eff = 0.29;
        if (lam > 4.60): eff = 0.27;
        if (lam > 4.70): eff = 0.24;
        if (lam > 4.80): eff = 0.22;
        if (lam > 4.90): eff = 0.19;
        if (lam > 4.98): eff = 0.08;

    if (strcmp(mode, "nsp") == 0):  # NIRSpec prism R=100
        tel_area = JWSTAREA  # using JWST telescope
        # eff = 0.95 * 0.95;  double pass, constant efficieincy until I find out better
        # eff = eff * 0.9 * 0.58 * 0.85;  tele, optics, detector
        # if (lam < 1.0)
        # eff = eff * (1.0 - 1.6*(1.0 - lam));   JWST + NIRSpec eff drops 2x between 1.0 & 0.5 microns
        eff = 0.16  # all eff values from grism plot on http://www.cosmos.esa.int/web/jwst/nirspec-pce */
        if (lam > 0.65): eff = 0.25;
        if (lam > 0.70): eff = 0.31;
        if (lam > 0.90): eff = 0.35;
        if (lam > 1.00): eff = 0.39;
        if (lam > 1.10): eff = 0.41;
        if (lam > 1.20): eff = 0.42;
        if (lam > 1.30): eff = 0.43;
        if (lam > 1.40): eff = 0.42;
        if (lam > 1.50): eff = 0.415;
        if (lam > 1.60): eff = 0.42;
        if (lam > 1.80): eff = 0.425;
        if (lam > 2.00): eff = 0.44;
        if (lam > 2.20): eff = 0.45;
        if (lam > 2.40): eff = 0.46;
        if (lam > 2.50): eff = 0.48;
        if (lam > 3.00): eff = 0.46;
        if (lam > 3.10): eff = 0.45;
        if (lam > 3.30): eff = 0.46;
        if (lam > 3.40): eff = 0.47;
        if (lam > 3.90): eff = 0.46;
        if (lam > 4.20): eff = 0.45;
        if (lam > 4.70): eff = 0.44;
        if (lam > 4.90): eff = 0.43;

    if (strcmp(mode, "ns1") == 0):  # NIRSpec R=1000 grating 1
        tel_area = JWSTAREA  # using JWST telescope
        eff = 0
        if (lam < 1.3):  # NIRSPEC gratings high eff direct ruled gold : SPIE 7731 2010
            eff = 0.3 + (lam - 1.0) * 2
        else:
            eff = 0.9 + (1.3 - lam) * 0.8
        eff = eff * 0.9 * 0.58 * 0.85 * 0.8  # tele, optics, filter, detector

    if (strcmp(mode, "ns2") == 0):  # NIRSpec R=1000 grating 2
        tel_area = JWSTAREA  # using JWST telescope
        eff = 0
        if (lam < 2.2):  # NIRSPEC gratings high eff direct ruled gold : SPIE 7731 2010
            eff = 0.3 + (lam - 1.7) * 1.2
        else:
            eff = 0.9 + (2.2 - lam) * 0.6
        eff = eff * 0.9 * 0.58 * 0.85 * 0.8  # tele, optics, filter, detector

    if (strcmp (mode, "ns3") == 0):  # NIRSpec R=1000 grating 3
        tel_area = JWSTAREA  # using JWST telescope
        eff = 0
        if (lam < 3.8):  # NIRSPEC gratings high eff direct ruled gold : SPIE 7731 2010
            eff = 0.3 + (lam - 2.9) * 0.65
        else:
            eff = 0.9 + (3.8 - lam) * 0.5
        eff = eff * 0.9 * 0.58 * 0.85 * 0.8  # tele, optics, filter, detector

    if (strcmp(mode, "lrs") == 0):  # MIRI LRS slitless mode
        tel_area = JWSTAREA  # using JWST telescope
        eff = 0
        # SLITLESS PCE from MIRI_FM_LRS_SLITLESS_PCE_06.00.00.fits,
        # MIRI-TN-00072_PCEs_Iss5.pdf, MIRI_slitless_LRS_PCE_AG_2016Jun.txt
        if (lam > 2.50): eff = 0.93 *  0.0005;
        if (lam > 2.75): eff = 0.93 *  0.0023;
        if (lam > 3.00): eff = 0.93 *  0.0030;
        if (lam > 3.25): eff = 0.93 *  0.0045;
        if (lam > 3.50): eff = 0.93 *  0.0102;
        if (lam > 3.75): eff = 0.93 *  0.0421;
        if (lam > 4.00): eff = 0.93 *  0.0507;
        if (lam > 4.25): eff = 0.93 *  0.0290;
        if (lam > 4.50): eff = 0.93 *  0.0287;
        if (lam > 4.75): eff = 0.93 *  0.0545;
        if (lam > 5.00): eff = 0.93 *  0.1216;
        if (lam > 5.25): eff = 0.93 *  0.2143;
        if (lam > 5.50): eff = 0.93 *  0.2668;
        if (lam > 5.75): eff = 0.93 *  0.2867;
        if (lam > 6.00): eff = 0.93 *  0.3068;
        if (lam > 6.25): eff = 0.93 *  0.3148;
        if (lam > 6.50): eff = 0.93 *  0.3345;
        if (lam > 6.75): eff = 0.93 *  0.3184;
        if (lam > 7.00): eff = 0.93 *  0.3086;
        if (lam > 7.25): eff = 0.93 *  0.3147;
        if (lam > 7.50): eff = 0.93 *  0.3226;
        if (lam > 7.75): eff = 0.93 *  0.3308;
        if (lam > 8.00): eff = 0.93 *  0.3378;
        if (lam > 8.25): eff = 0.93 *  0.3425;
        if (lam > 8.50): eff = 0.93 *  0.3481;
        if (lam > 8.75): eff = 0.93 *  0.3479;
        if (lam > 9.00): eff = 0.93 *  0.3468;
        if (lam > 9.25): eff = 0.93 *  0.3446;
        if (lam > 9.50): eff = 0.93 *  0.3384;
        if (lam > 9.75): eff = 0.93 *  0.3324;
        if (lam > 10.00): eff = 0.93 * 0.2949;
        if (lam > 10.75): eff = 0.93 * 0.2523;
        if (lam > 11.00): eff = 0.93 * 0.2199;
        if (lam > 11.25): eff = 0.93 * 0.1965;
        if (lam > 11.50): eff = 0.93 * 0.1795;
        if (lam > 11.75): eff = 0.93 * 0.1599;
        if (lam > 12.00): eff = 0.93 * 0.1401;
        if (lam > 12.25): eff = 0.93 * 0.1182;
        if (lam > 12.50): eff = 0.93 * 0.0942;
        if (lam > 12.75): eff = 0.93 * 0.0753;
        if (lam > 13.00): eff = 0.93 * 0.0610;
        if (lam > 13.25): eff = 0.93 * 0.0468;
        if (lam > 13.50): eff = 0.93 * 0.0364;
        if (lam > 13.75): eff = 0.93 * 0.0277;
        if (lam > 14.00): eff = 0.93 * 0.0209;
        if (lam > 14.25): eff = 0.93 * 0.0121;
        if (lam > 14.50): eff = 0.93 * 0.0032;
        if (lam > 14.75): eff = 0.93 * 0.0000;
        if (lam > 15.00): eff = 0.93 * 0.0000;

    if (strcmp(mode, "ost9") == 0):  # MISC slitless mode
        tel_area = OST9AREA  # using OST telescope
        eff = 0.0 
        # Use 3-bnd MISC transit spec Config 2, 4 lens, bare gold throughputs in MISC_transit_spec_efficiency_OST_concept2_TM_2018Jul22.pptx
        # Multiply by 0.9 for contamination / light loss, and 1.2 for AR coating the NEOCam detector
        if (lam > 2.85): eff = 0.9 * (0.218 + (lam-3.0)  * 0.157); #HgCdTe JWST
        if (lam >  4.0): eff = 0.9 * (0.375 - (lam-4.0)  * 0.0536); #HgCdTe JWST
        if (lam >  5.4): eff = 0.9 * 1.2 * (0.200 + (lam-5.4)  * 0.0604); #HgCdTe NEOCam
        if (lam >  7.8): eff = 0.9 * 1.2 * (0.345 - (lam-7.8)  * 0.0250); #HgCdTe NEOCam
        if (lam > 10.6): eff = 0.9 * (0.220 + (lam-10.6) * 0.0352); # Si:As JWST
        if (lam > 15.0): eff = 0.9 * (0.375 - (lam-15.0) * 0.0143); # Si:As JWST

    if (strcmp(mode, "ost6") == 0):  # MISC slitless mode
        tel_area = OST6AREA  # using OST telescope
        eff = 0.0
        # Use 3-bnd MISC transit spec Config 2, 4 lens, bare gold throughputs in MISC_transit_spec_efficiency_OST_concept2_TM_2018Jul22.pptx
        # Multiply by 0.9 for contamination / light loss, and 1.2 for AR coating the NEOCam detector
        if (lam > 2.85): eff = 0.9 * (0.218 + (lam-3.0)  * 0.157); #HgCdTe JWST
        if (lam >  4.0): eff = 0.9 * (0.375 - (lam-4.0)  * 0.0536); #HgCdTe JWST
        if (lam >  5.4): eff = 0.9 * 1.2 * (0.200 + (lam-5.4)  * 0.0604); #HgCdTe NEOCam
        if (lam >  7.8): eff = 0.9 * 1.2 * (0.345 - (lam-7.8)  * 0.0250); #HgCdTe NEOCam
        if (lam > 10.6): eff = 0.9 * (0.220 + (lam-10.6) * 0.0352); # Si:As JWST
        if (lam > 15.0): eff = 0.9 * (0.375 - (lam-15.0) * 0.0143); # Si:As JWST

    if (strcmp(mode, "ost5") == 0):  # MISC slitless mode
        tel_area = OST5AREA  # using OST telescope
        eff = 0.0
        # Use 3-bnd MISC transit spec Config 2, 4 lens, bare gold throughputs in MISC_transit_spec_efficiency_OST_concept2_TM_2018Jul22.pptx
        # Multiply by 0.9 for contamination / light loss, and 1.2 for AR coating the NEOCam detector
        if (lam > 2.85): eff = 0.9 * (0.218 + (lam-3.0)  * 0.157); #HgCdTe JWST
        if (lam >  4.0): eff = 0.9 * (0.375 - (lam-4.0)  * 0.0536); #HgCdTe JWST
        if (lam >  5.4): eff = 0.9 * 1.2 * (0.200 + (lam-5.4)  * 0.0604); #HgCdTe NEOCam
        if (lam >  7.8): eff = 0.9 * 1.2 * (0.345 - (lam-7.8)  * 0.0250); #HgCdTe NEOCam
        if (lam > 10.6): eff = 0.9 * (0.220 + (lam-10.6) * 0.0352); # Si:As JWST
        if (lam > 15.0): eff = 0.9 * (0.375 - (lam-15.0) * 0.0143); # Si:As JWST

    if (strcmp (mode, "g280") == 0):  # HST WFC3 UV grism mode
        tel_area = HSTAREA  # using HST telescope
        eff = 0
        if (lam <= 0.240):  # Get efficiencies from WFC3 Instrument handbook Chapter 8.2, Fig 8.2
            eff = -0.65 + (3.75 * lam)
        elif (lam <= 0.275):
            eff = 0.936 - (2.86 * lam)
        elif (lam <= 0.35):
            eff = 0.407 - (0.933 * lam)
        else:
            eff = 0.08
        eff = eff * 0.76  # telescope efficiency @ 300 nm = 0.87^2 - optics * det already included for WFC3
        eff = eff * 0.82 * 0.9  # fudge factor to match STScI WFC3 ETC
        # print("WFC3 G280 mode. lambda = %f   effic = %f\n" % (lam, eff))

    if (strcmp(mode, "dhs") == 0):  # JWST NIRCam DHS SW mode
        tel_area = JWSTAREA * 0.029  # JWST telescope area x relative pupil area of single DHS per Everett email
                                  # 2016 Jul 13 (see  DHS_performance_info_DK_2015Mar27.txt)
        eff = 0
        # DHS througputs tabulated in file NIRCam_DHS_blaze_throughput_efficiency.xlsx
        # - includes (incl. OTE + F150W2 filter: F150W2_nircam_plus_ote_throughput_moda_sorted.txt
        if (lam >= 1.00): eff = 0.093;
        if (lam >= 1.05): eff = 0.205;
        if (lam >= 1.10): eff = 0.241;
        if (lam >= 1.15): eff = 0.271;
        if (lam >= 1.20): eff = 0.169;  # Notch at 1.20 microns
        if (lam >= 1.25): eff = 0.312;
        if (lam >= 1.30): eff = 0.315;
        if (lam >= 1.35): eff = 0.330;
        if (lam >= 1.40): eff = 0.324;
        if (lam >= 1.45): eff = 0.202;  # Notch at 1.45 microns
        if (lam >= 1.50): eff = 0.342;
        if (lam >= 1.55): eff = 0.342;
        if (lam >= 1.60): eff = 0.332;
        if (lam >= 1.65): eff = 0.324;  # detector gap at 1.xx microns
        if (lam >= 1.70): eff = 0.317;
        if (lam >= 1.75): eff = 0.304;
        if (lam >= 1.80): eff = 0.296;
        if (lam >= 1.85): eff = 0.284;
        if (lam >= 1.90): eff = 0.270;
        if (lam >= 1.95): eff = 0.262;
        if (lam >= 2.00): eff = 0.250;
        if (lam >= 2.05): eff = 0.239;
        if (lam >= 2.10): eff = 0.234;

    if (strcmp(mode, "g141") == 0):  # HST WFC3 IR grism mode
        tel_area = HSTAREA  # using HST telescope
        eff = 0
        """
        if (lam <= 1.4):
            eff = 0.3 + 0.6 * (lam - 1.1)
        else:
            eff = 0.48
        """
        # Get efficiencies from http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2011-05.pdf Fig 4
        eff = 0.2
        if (lam > 1.10): eff = 0.32;
        if (lam > 1.15): eff = 0.37;
        if (lam > 1.20): eff = 0.42;
        if (lam > 1.30): eff = 0.46;
        if (lam > 1.40): eff = 0.48;
        if (lam > 1.50): eff = 0.46;
        if (lam > 1.60): eff = 0.43;
        if (lam > 1.65): eff = 0.20;
        if (lam > 1.67): eff = 0.10;
        if (lam > 1.69): eff = 0.005;

        # print("WFC3 G141 mode. lambda = %f   effic = %f\n" %(lam, eff))

    if (strcmp(mode, "finesse") == 0):  # FINESSE Explorer
        tel_area = FINESSEAREA  # using FINESSE telescope
        eff = 0.3  # tele, optics, gratings over 1 - 5um, detector

    lam = lam/1.0e4  # convert microns to cm; use cgs throughout
    eff = eff * 0.98  # cannot extract all pixels
    return(flux * t / (h * c / lam) * (lam / R) * eff * tel_area)

#elec(lam, R, s_flux, dtime, mode)
#print(elec(1.000100, 100.000000, 0.000097, 1.889989, "gr700xd"))






#------------------------------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------------------------------

if (len(argv) < 12):
    print("usage: pysyn_transit_spec.py Teff* logg* R_star(cm) K*(mag) p_infile itime(s) Nints R Mode precision_file outfile_prefix (i)\n")
    print("Modes: gr700xd = NIRISS grism: ncgsw = NIRCam SW grism: ncglw = NIRCam LW grism: nsp = NIRSpec prism: ns1 = NIRSpec grating1: ns2 = NIRSpec grating2: ns3 = NIRSpec grating3: lrs = slitless MIRI LRS; dhs = NIRCam DHS; g141 = HST G141 IR grism; finesse = FINESSE\n")
    print("Uses intrinsic instrument R if command line R is 0.\n")
    exit()

Teff_s = float(argv[1])  # effective Temp of star in K

logg_s = float(argv[2])  # log g of star cgs

s_radius = float(argv[3])  # stellar radius in cm

Kmag = float(argv[4])  # K mag of star

try:
    fin_p = open(argv[5], "r")
except:
    print("transit_spec: can't read %s\n" % argv[5])
    exit()


itime = float(argv[6])
if (itime < 0):
    print("transit_spec: minimum integration time is 0 s!\n")
    exit()

nints = int(argv[7])

R_command = float(argv[8])
# print("R_command = %5.0f\n" % R)

mode = str(argv[9])


try:
    fin_noise = open(argv[10], "r")
except:
    print("transit_spec: can't read %s\n" % argv[10])
    exit()

star_str = str(argv[11])
transit_str = str(argv[11])
div_str = str(argv[11])
star_str = star_str + ".star."
transit_str = transit_str + ".tr."
div_str = div_str + ".div."
star_str = star_str + str(argv[9])
transit_str = transit_str + str(argv[9])
div_str = div_str + str(argv[9])

# print("%s, %s, %s \n" % (argv[6], star, transit))

try:
    fout_s_sw = open(star_str, "w")
except:
    print("transit_spec: can't write %s.star\n" % argv[11])
    exit()

try:
    fout_tr_sw = open(transit_str, "w")
except:
    print("transit_spec: can't write %s.tr\n" % argv[11])
    exit()

try:
    fout_div = open(div_str, "w")
except:
    print("transit_spec: can't write %s.div\n" % argv[11])
    exit()

rad_factor = 1.0

"""
only open 1 wavelength file
strcpy( star, argv[9]);
strcpy( transit, argv[9]);
strcat( star, ".star.lw");
strcat( transit, ".tr.lw");

if ((fout_s_lw = fopen( star, "w")) == NULL) {
    printf("transit_spec: can't write %s.star.lw\n", argv[7]);
    exit(0);
}

if ((fout_tr_lw = fopen( transit, "w")) == NULL) {
    printf("transit_spec: can't write %s.tr.lw\n", argv[7]);
    exit(0);
}
"""

s_model = [] # star model
p_model = [] # planet model
precis = [] # noise model

# compute synthetic Phoenix stellar spectrum using pysynphot
Fe_H = 0.0 # assume solar metalicity

star = S.Icat('phoenix', Teff_s, Fe_H, logg_s)  # Nextgen model
star_norm = star.renorm(Kmag, 'vegamag', S.ObsBandpass('johnson,k'))

star_norm.convert('Micron')
star_norm.convert('flam')  # flam units: erg/s/cm^2/Ang

Npoints = len(star.wave)
for i in range(0, Npoints):
    if (star_norm.wave[i] < 0.5):
        start = i
    if (star_norm.wave[i] <= 28.5):
        stop = i

for i in range(start, stop):
    wav = star_norm.wave[i] 
    F = star_norm.flux[i] * 1.E8  #convert F* (ergs/s/cm2/A) to ergs/s/cm**2/cm
    s_model.append(Star(wav, F))
    
#ns_points = stop - start + 1
ns_points = len(s_model)
#print("Size of star model array = %d" %(ns_points))
s_model.sort(key=operator.attrgetter("wav"))

#print("index %d: wav: %f ---- F: %f" %(i, s_model[i-1].wav, s_model[i-1].F))
#print(s_model[i-10:])
#test = 3.61564689143e-10 * 2.998e10 / (995.405417352 * 995.405417352 * 1E-8)
#print(test)
# read model planet wavelengths & radii from the text file given as arg2, put into 'p_model'
i = 0
for line_p in fin_p:
    p = line_p.split()
    p[0] = float(p[0])
    p[1] = float(p[1])
    wav = p[0]  # file wavelength already in microns
    # p_model[i].rad = p[1] * 6.378e8; convert J. F. GJ 1214b Earth radius to cm
    rad = sqrt(p[1]) * s_radius  # convert T Kataria area ratio to planet radius
     # p_model[i].rad =p[1] * 1e5 * rad_factor; convert km to cm and scale if ingress / egress
     # if ( p_model[i].rad < 1e5) printf("%9f\n", p_model[i].rad);
    p_model.append(Planet(wav,rad))

    i += 1
fin_p.close()
np_points = i

# sort the p_model into wavelength order
#print(p_model[:10])
#print("transit_spec: Sorting planet spectrum...\n")
p_model.sort(key=operator.attrgetter("wav"))
#print("index %d: wav: %f ---- rad: %f" %(i, p_model[i-1].wav, p_model[i-1].rad))

#print(p_model[:10])
# read noise wavelengths & radii from the text file given as arg8, put into 'precis'
#x = [Planet(5,7), Planet(1,2)]
#x.sort(key=operator.attrgetter("wav"))
#print(x)
i = 0
for line_n in fin_noise:
    p = line_n.split()
    wav = float(p[0])  # already in um
    pre = float(p[1])
    precis.append(Noise(wav, pre))
    i += 1

fin_noise.close()
n_noise = i
#print("index %d: wav: %f ---- pre: %f" %(i, precis[i-1].wav, precis[i-1].pre))

# print("lambda_long = %f \n" % (p_model[np_points-1].wav))
if (strcmp(mode, "gr700xd") == 0):
    print("transit_spec: NIRISS R=700 x-disp Grism mode: *** Using lower of native resolution and command line R\n")
    # NIRISS wavelength limits
    # sw1 = 0.60;  std mode x-disp m = 1 only
    sw1 = 0.90  # bright mode m = 1 only
    sw2 = 2.499
    R_intr = 700.0
    pxl_area = 0.065 * 0.065  # sq arcsec pixel area
    # NIRCam backgrounds: F090 = 152, F150W2 = 231, F200W = 187 e/s/sq arcsec
    bkg_elec = 450 * pxl_area
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "ncgsw") == 0):
    print("transit_spec: NirCam Short-Wave Grism mode...\n")
    # NIRCam wavelength limits
    sw1 = 1.0
    sw2 = 2.0
    R_intr = 750.0
    pxl_area = 0.032 * 0.032  # sq arcsec pixel area
    # NIRCam backgrounds: F150W2 = 231 e/s/sq arcsec
    bkg_elec = 231 * pxl_area
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp (mode, "ncglw") == 0):
    print("transit_spec: NirCam Long-Wave Grism mode: *** Using lower of native resolution and command line R\n")
    # NIRCam wavelength limits
    # lw1 = 2.42; true SW cuton - dichroics
    lw1 = 2.43  # Use NIRCam 2.42 - 2.499 microns
    lw2 = 4.999
    sw1 = lw1
    sw2 = lw2
    R_intr = 1700.0
    pxl_area = 0.064 * 0.064  # sq arcsec pixel area
    # NIRCam backgrounds: F277W2 = 106, F356W = 94, F444W = 308 e/s/sq arcsec
    bkg_elec = 308 * pxl_area *0.7  # attenuate bkg by grism transmission
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "nsp") == 0):
    print("transit_spec: NIRSpec prism mode: *** Using lower of native resolution and command line R \n");
    # NIRSpec prism wavelength limits
    sw1 = 0.7
    sw2 = 4.999
    R_intr = 0.0  # variable intrinsic R - gets set later for each wavelength
    pxl_area = 0.110 * 0.110  # sq arcsec pixel area
    bkg_elec = 1.0  # little background through 1.6 arcsec slit
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp (mode, "ns1") == 0):
    print("transit_spec: NIRSpec R=1000 Grating 1 mode...\n")
    # NIRSpec grating 1 wavelength limits
    # R = 1000.0; temporarily set R in command line
    sw1 = 1.0
    sw2 = 1.8
    R_intr = 1000.0
    bkg_elec = 1.0  # little background through 1.6 arcsec slit
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "ns2") == 0):
    print("transit_spec: NIRSpec R=1000 Grating 2 mode...\n")
    # NIRSpec grating 2 wavelength limits
    # R = 1000.0; */  /* temporarily set R in command line
    sw1 = 1.7
    sw2 = 3.0
    R_intr = 1000.0
    bkg_elec = 1.0  # little background through 1.6 arcsec slit
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "ns3") == 0):
    print("transit_spec: NIRSpec R=3000 Grating 3 mode...\n")
    # NIRSpec grating 3 wavelength limits
    sw1 = 2.9
    sw2 = 5.0
    R_intr = 1000.0
    bkg_elec = 1.0  # little background through 1.6 arcsec slit
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "lrs") == 0):
    print("transit_spec_noise: *** Using lower of native resolution and command line R\n")
    # MIRI LRS slitless wavelength limits - need to get better eff data longer than 10 um
    sw1 = 5.0
    sw2 = 13.0
    R_intr = 0.0  # variable intrinsic R - gets set later for each wavelength
    pxl_area = 0.11 * 0.11  # sq arcsec pixel area
    # LRS BKG ~ F560W + F770W + F1000W + F1130W + 0.3 F1280W = 204 + 1960 + 2962 + 1243 + 0.3*5183 e/s/sq arcsec
    # so LRS BKG ~ 7923 e/s/sq arcsec; above values from STScI ETC
    # bkg_elec = 7923 * pxl_area; That is 96e- /sec / pixel; I get 169 e-/s/pixel from integrating over bandpass in
    # MIRI_LRS_background.ipynb : see ~/WD/NGST/MIST/LRS_backgrounds.txt
    bkg_elec = 169.0
    # JWST fast steering mirror noise equivalent bandwidth = 8 Hz

if (strcmp(mode, "ost9") == 0) or (strcmp(mode, "ost6") == 0) or (strcmp(mode, "ost5") == 0):
    print("transit_spec_noise: *** Using lower of native resolution and command line R\n")
    # OST2 MISC wavelength limits 
    sw1 = 3.0
    sw2 = 20.5
    R_intr = 300  # variable intrinsic R - gets set later for each wavelength
    field_area = PI * 3.0 * 3.0 / (206265.)**2 # str field  area of 3 arcsec radius field aperture

if (strcmp(mode, "g280") == 0):
    print("transit_spec: HST WFC3 G280 R=70 UV 200-400 nm Grism...\n")
    # HST WFC3 UV grism wavelength limits
    sw1 = 0.20
    sw2 = 0.40
    R_intr = 70.0
    # HST has no fast steering mirror; use the input noise floor values

if (strcmp(mode, "dhs") == 0):
    print("transit_spec: DHS  1.0 - 2.x um Grism...\n")
    # HST WFC3 UV grism wavelength limits
    sw1 = 1.00  # from JWST-STScI-001990 Fig 4 and ~/WD/NGST/NIRCam/WFS/DHS_SCA_projection_0deg.jpg
    sw2 = 2.0  # From F150W2 cutoff and ~/WD/NGST/NIRCam/WFS/DHS_SCA_projection_0deg.jpg
               # - longer wavelengths are aliased with m=2 lambda/2
    pxl_area = 0.032 * 0.032  # NIRCam SW sq arcsec pixel area */
    # WHST WFC3 avg background is 1.3 e-/s/pixel: http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2011-01.pdf
    bkg_elec = 0.24
    # NIRCam backgrounds: F150W2 = 231 e/s/sq arcsec

if (strcmp(mode, "g141") == 0):
    print("transit_spec: HST WFC3 G141 R=150 1.10 - 1.70 um Grism...\n")
    # HST WFC3 UV grism wavelength limits
    sw1 = 1.1
    sw2 = 1.7
    R_intr = 150.0  # 2 pixels at 1.41 um
    pxl_area = 0.121 * 0.121  # sq arcsec pixel area
    # WHST WFC3 avg background is 1.3 e-/s/pixel: http://www.stsci.edu/hst/wfc3/documents/ISRs/WFC3-2011-01.pdf
    bkg_elec = 1.3 # attenuate bkg by grism transmission
    # use the input noise floor values for G141 fast scan mode 5"/sec (McCullough & Mackenty 2012

if (strcmp(mode, "finesse") == 0):
    print("transit_spec_noise: FINESSE Explorer... \n")
    sw1 = 0.7
    sw2 = 5.0
    R_intr = 1000.0
    # FINESSE fast steering mirror noise equivalent bandwidth = 8 Hz

# print("transit_spec: Note - NOT adding Photon noise!\n")

print("transit_spec_noise: Note - Photon noise _IS_ added!\n")
# ntime10 = (itime + 1) / 10; */ /* break into units of 10s to not exceed long 2E9 for Poisson noise
# d_itime = itime - (ntime10 * 10.0)
# print("transit_spec_noise: itime = %f\n" % (10.0 * ntime10 + d_itime))
# Above commented out because now break exposures into nints * itime/nints

if (p_model[0].wav <= sw1):  # check if planet model data goes down to SW limit of observing mode
    lam = sw1
else:
    print("transit_spec_noise: shortest planet spectrum wavelength is greater than selected mode SW limit. Reseting SW limit to %7.4f\n" % p_model[0].wav)
    lam = p_model[0].wav

if (p_model[np_points-1].wav < sw2):  # Reset mode's LW limit if planet model wavelengths don't go that long
    sw2 = p_model[np_points-1].wav
    print("transit_spec_bb_noise: longest planet spectrum wavelength is less than selected mode LW limit. Reseting LW limit to %7.4f\n" % p_model[np_points-1].wav)

#print(lam)
#print(sw2)
while (lam < sw2):
    R = R_command

    if (strcmp(mode, "gr700xd") == 0):  # Set R as fn of lam for NIRISS GR700XD
        if (lam > 0.9):
            R_intr = 700 * lam / 1.350  # m = 1 values from Loic Albert April 2014
        else:
            R_intr = 700 * lam / 0.675  # m = 2 values from Loic Albert April 2014
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        npix = 2 * 40 * R_intr/R  # spectral x spatial
        det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints)

    if (strcmp(mode, "lrs") == 0):  # Set R as fn of lam for MIRI LRS
        R_intr = 40 + 24 * (lam - 5)
        # LRS resolution is set by PSF for all wavelengths lam < 14 um: 5 pixel wide slit: 2.33 pxl is 6.8 um l/D
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        npix = 2 * 10 * R_intr/R  # spectral x spatial
        if (lam > 7.0):
            npix = 2 * (lam/7.0) * 10 * R_intr/R  # spectra x spatial
        det_noise = CDS_SiAs * sqrt(npix) * sqrt(nints)

    if (strcmp(mode, "ost9") == 0) or (strcmp(mode, "ost6") == 0) or (strcmp(mode, "ost5") == 0): 
        if (strcmp(mode, "ost9") == 0) : tel_area = OST9AREA  # using OST telescope
        if (strcmp(mode, "ost6") == 0) : tel_area = OST6AREA  # using OST telescope
        if (strcmp(mode, "ost5") == 0) : tel_area = OST5AREA  # using OST telescope
        
        R_intr = 300.0
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        if (lam < 5.4):
            npix = 2 * 12 * R_intr/R  * NPUPIL # spectral x spatial x #of pupil slices
        else:
            npix = 4 * 22 * R_intr/R  * NPUPIL # spectral x spatial x #of pupil slices
            
        if (lam > 2.85): 
            zodi = 0.35E6 * 1E-26 * 1E3 #0.35 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_HgCdTe_5um # 1.3 for more than minimum, PCE ~ 0.3
        if (lam >  6.0): 
            zodi = 1.0E6 * 1E-26 * 1E3 #1.0 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_HgCdTe_10um # 1.3 for more than minimum, PCE ~ 0.3        
        if (lam >  8.0): 
            zodi = 3.0E6 * 1E-26 * 1E3 #3.0 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_HgCdTe_10um # 1.3 for more than minimum, PCE ~ 0.3        
        if (lam > 10.0): 
            zodi = 8.5E6 * 1E-26 * 1E3 #8.5 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_SiAs # 1.3 for more than minimum, PCE ~ 0.3        
        if (lam > 11.0): 
            zodi = 11E6 * 1E-26 * 1E3 #8.5 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_SiAs # 1.3 for more than minimum, PCE ~ 0.3        
        if (lam > 14.0): 
            zodi = 14E6 * 1E-26 * 1E3 #8.5 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_SiAs # 1.3 for more than minimum, PCE ~ 0.3        
        if (lam > 17.0): 
            zodi = 17E6 * 1E-26 * 1E3 #8.5 MJy/str Glasse+ 2015 A+B converted to erg/cm^2/Hz 
            bkg_elec = 1.3 * zodi * c/((lam*1E-4)**2) * tel_area * lam*1E-4/(h * c) * field_area / npix * 0.3 * lam/R_intr * 1E-4 + DC_SiAs # 1.3 for more than minimum, PCE ~ 0.3        
        #print("lam = %4.1f, npix = %4.0f, background/pxl = %f" % (lam, npix, bkg_elec))
        if (lam < 10.5):
            det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints) / 2.0 #assume mult reads
        else:
            det_noise = CDS_SiAs * sqrt(npix) * sqrt(nints) / 2.0 #assume mult reads

    if (strcmp (mode, "ncglw") == 0):  # Set R as fn of lam for NIRCam grisms
        R_intr = 1700 * lam / 4.0  # constand dispersion grating
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        npix = 2 * 15 * R_intr/R  # spectral x spatial
        det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints)

    if (strcmp (mode, "nsp") == 0):  # Set R as fn of lam for NIRSpec prism
        R_intr = 1.6567*pow(lam, 4.0) - 21.179*pow(lam, 3.0) + 108.14*pow(lam, 2.0) - 189.86*lam + 135.16
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        npix = 2 * 10 * R_intr/R  # spectral x spatial
        det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints)

    if (strcmp(mode, "dhs") == 0):  # Set R as fn of lam for NIRCam DHS
        Rnative = 2344 * (lam / 1.36)
        # 2 pixel dispersion of DHS is 0.580 nm per ~/WD/NGST/NIRCam/WFSDHS_performance_info_DK_2015Mar27.txt
        if ((Rnative < R) or (R < 0.01)):
            R = Rnative  # use native R if it is less than command line R
        npix = 2 * 30 * Rnative / R  # spectral x spatial - Assume ~30 spatial pixels per DHS extraction for 3.5 pxl FWHM PSF, no tilt
        det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints)

    if (strcmp(mode, "g141") == 0):  # Set R as fn of lam for HST WFC3 G141
        R_intr = 150 * (lam / 1.41)
        # 2 pixel dispersion of HST WFC3 G141 is 9.4 nm
        if ((R_intr < R) or (R < 0.01)):
            R = R_intr  # use native R if it is less than command line R
        npix = 2 * 40 * R_intr/R  # spectral x spatial - scan mode is 40 pixels high
        det_noise = CDS_HgCdTe * sqrt(npix) * sqrt(nints)

    # print("lambda = %f  R = %f  R_intr = %f\n" % (lam, R, R_intr))

    if (R_command < 0.1):
        R = R_intr  # use intrinsic R if command line R is 0

    dlam = lam/R

    precision = 1.0
    delta = 10.0  # start with high difference, 10 um
    i = 0
    while i < n_noise:
        if (fabs(precis[i].wav - lam) < delta):
            precision = precis[i].pre
            delta = fabs(precis[i].wav - lam)
        i += 1

    # print("lambda = %f  SNR = %f\n" %(lam, maxsnr))
    # maxsnr = maxsnr * sqrt(itime * fsm_bw) * sqrt(R_intr/R) adjust noise floor for FSB BW and final R
    # maxsnr = maxsnr * sqrt(R_intr/R) adjust noise floor for final R

    s_flux = 0.0
    j = 0
    i = 0
    while i < ns_points:
        if (fabs(s_model[i].wav - lam) <= dlam/2):
            s_flux = s_model[i].F + s_flux
            j += 1
        i += 1

    # print(s_flux, end="--")

    if (j == 0):
        dlam_star = fabs(lam - s_model[0].wav)
        s_flux = s_model[0].F
        i = 0
        while i < ns_points:
            if (fabs(s_model[i].wav - lam) < dlam_star):
                s_flux = s_model[i].F
                dlam_star = fabs(s_model[i].wav - lam)
                s_index = i
            i += 1
        # print("lambda = %8.4e index = %d star flux = %7.4e %7.4e\n" %(lam, s_index, s_model[s_index].F, s_flux))
        if (s_index < ns_points):
            if (dlam_star > (2*dlam)):
                print("transit_spec: WARNING: Selected R is MUCH higher than stellar model\n")
            else:
                print("transit_spec: WARNING: Selected R is higher than stellar model\n")
            j = 1

        else:
            print("transit_spec: Wavelength gtr than stellar model cutoff - aborting\n")
            exit()


    s_flux = s_flux / j
    #print(s_flux, end="--")
    # print(j, end="--")

    # s_flux = s_flux  * (s_radius * s_radius) /  (d * d)  # observed flux
    # s_flux = s_flux  * 4 * PI *  (s_radius * s_radius) /  (d * d) ;  observed flux for Kurucz

    # s_fluxBB = blam(lam, 5930.0) * PI * (s_radius * s_radius) /  (d * d)
    # print("lambda = %f s_flux = %5.2e BB_flux = %5.2e\n"  % (lam, s_flux, s_fluxBB))

    p_radius = 0.0
    j = 0
    i = 0
    while i < np_points:
        if (fabs(p_model[i].wav - lam) <= dlam/2):
        # print("%d %6f %6.4e\n" % (i, p_model[i].wav, lambda))
            p_radius = p_model[i].rad + p_radius
            j += 1
        i += 1

    if (j == 0):
        j = 1
        min = lam
        i = 0
        while i < np_points:
            if (fabs(p_model[i].wav - lam) <= min):
                p_radius = p_model[i].rad
                min = fabs(p_model[i].wav - lam)
            i += 1

        print("transit_spec: spectrum R >= planet model: lam, lam/diff, p_radius = %6.3f, %6.0f, %7.4e\n" % (lam, lam/min, p_radius))
    p_radius = p_radius / j

    nelec_noise = 0.0
    nelec_nonoise = 0.0
    bkg_noise = 0.0
    bkg_nonoise = 0.0
    dtime = itime / (nints*1.0)
    i = 0
    #print(bkg_elec, npix, dtime)
    #a = elec(lam, R, s_flux, dtime, mode)
    #print("%f--" % a, end="")
    # print("lambda: %f R: %f s_flux: %f dtime: %f mode: %s" % (lam, R, s_flux, dtime, mode))

    while i < nints:
        bkg   = bkg_elec * npix * dtime
        nelec = elec(lam, R, s_flux, dtime, mode) + bkg
        #if (i == 0):
        #    print("lambda: %f, R: %f, s_flux: %f, dtime: %f, mode: %s" %(lam, R, s_flux, dtime, mode))
        nelec_nonoise = nelec + nelec_nonoise
        bkg_nonoise = bkg + bkg_nonoise
        # NOTE RANLIB ignpoi routine to add Poisson noise is limited to long integers, 2.2E9
        # if (nelec  > 2.199E9 or nelec < 0 ):
        #    print("Error: Star #electrons gtr than 2E9! %6.3f %d %f %f %f\n" % (lam, nints, nelec, nelec_noise, dtime))
        # ignpoi -> np.random.poisson
        nelec_noise = np.random.poisson(nelec, size=None) + nelec_noise
        bkg_noise   = np.random.poisson(bkg, size=None) + bkg_noise
        i += 1
    #print("%f" % nelec, end = " ")
    #print("%f" % bkg_noise)
    # print("Flux = %6.4e,  noise = %6.4e  electrons\n" % (nelec, (nelec_noise-nelec)))
    # print("Flux = %8.5e, with noise = %8.5e electrons\n" % (nelec*itime/10.0, nelec_noise))
    # if (nelec_noise < 0) print("Less than Zero! %f %f\n" % (nelec, nelec_noise))

    # gennor -> np.random.normal
    nelec_noise = np.random.normal(nelec_noise, det_noise, size=None)  # add read noise
    # nelec_noise = gennor(nelec_noise, nelec_noise * precision)  do not add noise floor yet
    bkg_noise = np.random.normal(bkg_noise, det_noise, size=None)  # add read noise

    s_elec = nelec_noise  # includes background
    s_elec_nonoise = nelec_nonoise  # includes background
    s_noise = sqrt(s_elec_nonoise + (s_elec_nonoise * precision)*(s_elec_nonoise * precision) + (det_noise * det_noise))  # add floor just for writing out the value to the file
    bg_noise = sqrt(bkg_nonoise + (det_noise * det_noise))


    fout_s_sw.write("%6f %8.6e %8.6e %8.6e %4.1f\n" % (lam*1E4, (s_elec-bkg_noise), (s_elec_nonoise-bkg_nonoise), s_noise, npix)) # in Angstroms star file, includes #extraction pixels
    # now do in-transit flux
    # recompute # of electrons from star and recompute its photon noise because measurements
    # are made separately: Ranlib should generate new noise values every time noise functions
    # are called, based on a unique initial seed

    elec_sum = 0.0
    bkg_noise = 0.0
    bkg_nonoise = 0.0
    nelec_noise = 0.0
    nelec_nonoise = 0.0
    dtime = itime / (nints*1.0)

    i = 0
    while i < nints:
        nelec = elec(lam, R, s_flux, dtime, mode)
        if nelec < 1:
            print("lam = %f, R = %f, s_flux = %f, dtime = %f, mode = %s" % (lam, R, s_flux, dtime, mode))
        # nelec_noise = (double) ignpoi((float) nelec);  Don't need to cast types explicitly
        # NOTE RANLIB ignpoi routine to add Poisson noise is limited to long integers, 2.2E9
        # if (nelec  > 2.199E9 or nelec < 0 ):
        #   print("Error: Transit #electrons gtr than 2E9! %f %f\n" % (nelec, nelec_noise))
        elec_sum = elec_sum + nelec
        bkg   = bkg_elec * npix * dtime
        nelec = (nelec * (1 - (p_radius * p_radius / (s_radius * s_radius)))) + bkg
        nelec_nonoise = nelec + nelec_nonoise
        bkg_nonoise = bkg + bkg_nonoise
        nelec_noise = np.random.poisson(nelec, size=None) + nelec_noise
        bkg_noise   = np.random.poisson(bkg, size=None) + bkg_noise
        i += 1

    nelec_noise = np.random.normal(nelec_noise, det_noise, size=None)  # add read noise
    # nelec_noise = gennor(nelec_noise, nelec_noise * precision)  do not add systematic noise yet

    tr_elec = nelec_noise  # includes background
    tr_elec_nonoise = nelec_nonoise  # includes background
    tr_noise = sqrt(tr_elec_nonoise + (det_noise * det_noise))
    s_noise = sqrt(s_elec_nonoise + (det_noise*det_noise))

    s_tr = s_elec_nonoise - tr_elec_nonoise
    s = s_elec_nonoise - bkg_nonoise
    div = s_tr/s  # no noise
    div_noise = (s_elec - tr_elec)/(s_elec - bkg_noise)
    # div_noise = gennor(div_noise, div_noise * precision)
    div_noise = np.random.normal(div_noise, precision)  # correct
    sigma_s_tr = sqrt((tr_noise * tr_noise) + (s_noise * s_noise))
    sigma_s = sqrt((s_noise * s_noise) + (bg_noise * bg_noise))
    sigma_div = sqrt(pow((sigma_s_tr / s),2) + pow((s_tr/(s * s)* sigma_s),2) + (1.0 * precision * precision))  # 1.0 means divided value has input precision

    # sigma_div = (tr_elec_nonoise / s_elec_nonoise) * sqrt(pow(s_noise/s_elec_nonoise, 2) + pow(tr_noise/tr_elec_nonoise, 2)) incorrect value!

    # now add noise floor to star and transit, just for output files
    tr_elec = np.random.normal(tr_elec, tr_elec * precision)
    tr_noise = sqrt(tr_elec_nonoise + (tr_elec_nonoise * precision)*(tr_elec_nonoise * precision) + (det_noise * det_noise))

    fout_tr_sw.write("%6f %8.6e %8.6e %8.6e\n" % (lam, tr_elec, tr_elec_nonoise, tr_noise)) # write lamda in microns

    fout_div.write("%8.5f % 8.6f %8.6f %8.6f\n" % (lam, div_noise, div, sigma_div))
    lam = lam + dlam

print("output format: lambda (microns)  flux_or_ratios   noiseless_value   1_sigma_noise\n")
print("\n")
fout_s_sw.close()
fout_tr_sw.close()
fout_div.close()


#!/usr/bin/env python

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

nc_lrsclear = np.loadtxt("%s.div.%s" % (argv[11], mode)) #usecols=(0,2,3))
#print("%s.div.%s" % (argv[10], mode))

fig1, ax1 = plt.subplots(figsize=[15,8])
minorLocator = AutoMinorLocator()
ax1.xaxis.set_minor_locator(minorLocator)
minorLocator = AutoMinorLocator()
ax1.yaxis.set_minor_locator(minorLocator)

ax1.xaxis.set_ticks_position('both')
ax1.yaxis.set_ticks_position('both')
ax1.tick_params(axis='y',which="minor", direction="in")
ax1.tick_params(axis='y',which="major", direction="in")
ax1.tick_params(axis='x',which="minor", direction="in")
ax1.tick_params(axis='x',which="major", direction="in")

plt.plot(nc_lrsclear[:,0], nc_lrsclear[:,2]*1E6, color='blue')
plt.errorbar( nc_lrsclear[:,0], nc_lrsclear[:,1]*1E6, nc_lrsclear[:,3]*1E6, color='black', ecolor='black', fmt='o', capsize=4)

# please change to auro x axis or else set for each mode
#plt.xlim([2.40, 5.0])
#plt.ylim([0.0136, 0.0151])
#plt.ylim([5350, 5600])
#plt.xscale('symlog')
#ax1.set_xticklabels([1,2,3,4,5,7,10]) # need both xticklabels and xticks
#ax1.set_xticks([1,2,3,4,5,7,10])  # need both xticklabels and xticks


plt.title("%s %s" % (argv[11], mode))
plt.xlabel("Wavelength ($\mu$m)", fontsize=14)
plt.ylabel("Absorption Depth $(R_p/R_*)^2$ (ppm)", fontsize=14)

#plt.show()
plt.savefig("%s_%s.png" % (argv[11], mode))
