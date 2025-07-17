# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:44:53 2024

@author: m.lagoin
"""
# %% Pour Julie

from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3
import PIL
from PIL import Image
import shutil
import glob
from scipy.stats import norm
import imageio
import os
import cv2
import time
import random
import pims
import math
import pickle
import multiprocessing

from scipy.optimize import curve_fit
import trackpy as tp
from pandas import DataFrame, Series  # for convenience
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.io import savemat



exec(open(r"C:\Users\m.lagoin\Documents\Python Scripts\Marc_params.py").read())

# Optionally, tweak styles.
mpl.rc('figure',  figsize=(10, 5))
mpl.rc('image', cmap='gray')


@pims.pipeline
def gray(image):
    # Take just the green channel
    if len(np.shape(image)) > 2:
        return 0.2126*image[:, :, 0] + 0.7152*image[:, :, 1] + 0.0722*image[:, :, 2]
    else:
        return image[:, :]

def compute_increments_for_all_scales(data_,fps_):
    max_scale_=int(np.floor(np.log(len(data_))/np.log(2)));
    intermediate_list=[ 0.1*n for n in range(1,10*max_scale_,1)]
    scales_list_=np.unique(np.floor([ 2**n for n in intermediate_list])).astype('int')
    
    increments_all_scales_=np.nan*np.zeros(shape = (len(scales_list_),len(data_)));
    
    for kk in range(len(scales_list_)):
        inc_=data_[scales_list_[kk]+1:]-data_[1:-scales_list_[kk]];
        increments_all_scales_[kk,0:len(inc_)]=inc_[:];
        del inc_
    
    scales_list_in_points_=scales_list_;
    scales_list_=[1/fps_*n for n in scales_list_];
    return scales_list_, increments_all_scales_ , scales_list_in_points_

 
# %% Code 1 image processing on .tif ; save on .mat and .csv


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\30_10_2024\beads_in_chanel_A_130mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\30_10_2024\beads_in_chanel_A_130mbar_1\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\30_10_2024\beads_in_chanel_A_130mbar_2\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\30_10_2024\beads_in_chanel_A_230mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\30_10_2024\beads_in_chanel_A_330mbar\TIF"


fec = 100 # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel



# % PUCE A
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\31_10_2024\beads_plus_Chlamy_in_chanel_A_30mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\31_10_2024\beads_plus_Chlamy_in_chanel_A_60mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\31_10_2024\beads_plus_Chlamy_in_chanel_A_90mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\31_10_2024\beads_plus_Chlamy_in_chanel_A_120mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\31_10_2024\beads_plus_Chlamy_in_chanel_A_150mbar\TIF"



fec = 100. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel
# U_typical=864./25 # en pixel par frame
# # P_typical=120 #○ en mbar

# # %07/11/2024
# # Puce_B
# # Broad

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\Proad\beads_plus_chlamy_in_chanel_B_33mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\Proad\beads_plus_chlamy_in_chanel_B_63mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\Proad\beads_plus_chlamy_in_chanel_B_93mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\Proad\beads_plus_chlamy_in_chanel_B_123mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\Proad\beads_plus_chlamy_in_chanel_B_153mbar\TIF"


fec = 130. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel

# %07/11/2024
# Puce_B
# précis

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_33mbar_cropped\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_63mbar_cropped\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_93mbar_cropped\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_123mbar_cropped\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_153mbar_cropped\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\07_11_2024\precis\beads_plus_chlamy_in_chanel_B_183mbar_cropped\TIF"


# fec = 264. # acquisition frequency 
# echelle=500e-6/1400 #scale of one pixel




target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\19_11_2024\beads_3mum_in_chanel_A_30mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\19_11_2024\beads_3mum_in_chanel_A_45mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\19_11_2024\beads_3mum_in_chanel_A_60mbar\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\19_11_2024\beads_3mum_in_chanel_A_75mbar\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\19_11_2024\beads_3mum_in_chanel_A_90mbar\TIF"


fec = 130. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\22_11_2024\Chalmy_in_chanel_B_sin_0mbar_0p0s\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\22_11_2024\Chalmy_in_chanel_B_sin_5mbar_0p5s\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\22_11_2024\Chalmy_in_chanel_B_sin_5mbar_1s\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\22_11_2024\Chalmy_in_chanel_B_sin_5mbar_5s\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\22_11_2024\Chalmy_in_chanel_B_sin_5mbar_10s\TIF"

fec = 40. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel





# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_10mbar_1s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_triangle_10mbar_5s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_triangle_10mbar_5s_1\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_0mbar_10s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_2mbar_1s\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_5mbar_1s\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_5mbar_1s\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_2mbar_1s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_2mbar_10s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_4mbar_10s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_5mbar_2s\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_5mbar_10s\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\27_11_2024\PUCE_C\Chalmy_in_chanel_C_sin_10mbar_10s\TIF"


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_0mbar_1s_ultra\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_0mbar_1s_ultra_1\TIF"


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_1\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_ultra\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_4\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_3\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_2\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\CAMY\28_11_2024\PUCE_C\Chalmy_in_chanel_B_tri_2mbar_1s_ultra_1\TIF"

fec = 40. # acquisition frequency 

echelle=500e-6/1400 #scale of one pixel






target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_1mbar_10s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_1mbar_10s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_1mbar_5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_1mbar_10s_80fps_after_shaking\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_2p5mbar_5s_80fps_after_shaking\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_5mbar_1s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_5mbar_2p5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\17_12_2024\Marc\Chalmy_in_chanel_C_sine_10mbar_0p5s_80fps\TIF"
fec = 80. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_1mbar_5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_1mbar_5s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_1mbar_2p5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_1mbar_2p5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_1mbar_1s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_2p5mbar_5s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_2p5mbar_5s_80fps_post\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_2p5mbar_10s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_2p5mbar_10s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_1s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_1s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_2p5s_80fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_2p5s_80fps_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_5s_80fps\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\18_12_2024\Chalmy_in_chanel_C_sinei_5mbar_5s_80fps_1\TIF"


fec = 80. # acquisition frequency 
echelle=500e-6/1400 #scale of one pixel





target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\16_01_2025\Chalmy_in_chanel_C_sinei_0mbar_0s_40fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\16_01_2025\Chalmy_in_chanel_C_sinei_1mbar_1s_40fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\16_01_2025\Chalmy_in_chanel_C_sinei_2mbar_1s_40fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\16_01_2025\Chalmy_in_chanel_C_sinei_4mbar_1s_40fps\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\16_01_2025\Chalmy_in_chanel_C_sinei_8mbar_1s_40fps\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\20_01_2024\Chalmy_in_chanel_C_sinei_0mbar_0s_40fps\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\04_02_2025\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x20_sinei_10mbar_1s_40fps_long\TIF"
echelle=500e-6/1400 #scale of one pixel


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\04_02_2025\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x05_sinei_10mbar_1s_40fps_long_1\TIF"
echelle=180e-6/100 #scale of one pixel


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long_after\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_2s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long_apres\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_1s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long_apres\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_0p5s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"


# Puce A
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long_apres\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_4s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_4s_40fps_long\TIF"


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long_apres\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_2s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long_after\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long_1\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_6mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_10mbar_1s_40fps_long\TIF"


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_0p5s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_0p5s_40fps_long\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_0p5s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_0p5s_40fps_long\TIF"



# %%

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_8s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_8s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_8s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_8s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_8s_40fps_long\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_4s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_4s_40fps_long\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_2s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_24mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_28mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_2s_40fps_long\TIF"


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_24mbar_1s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_28mbar_1s_40fps_long\TIF"
# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_1s_40fps_long\TIF"



target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_0p5s_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_0p5s_40fps_long\TIF"


target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_1\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_2\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_3\TIF"

target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2_\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__1\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__2\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__3\TIF"


echelle=180e-6/110 #scale of one pixel
fec = 40. # acquisition frequency 
# echelle=500e-6/1400 #scale of one pixel



savefolder=os.path.join(target_folder,"tracking_results")
if not os.path.exists(savefolder):
    os.makedirs(savefolder)
savefolder2=os.path.join(savefolder,r"tracking_parts")

if not os.path.exists(savefolder2):
    print(r"error, please track first")
# %

extension = 'csv'
os.chdir(savefolder2)
result = glob.glob('*.{}'.format(extension))
print('Number of parts found :')
print(len(result))

for index_slice in range(len(result)):#↓len(frames)//number_slices):
    fname_to_open=r"f_tot_part_"+ str(index_slice) +r".csv"
    link_path_=os.path.join(savefolder2, fname_to_open)
    f_to_add=pd.read_csv(link_path_)
    if index_slice==0:
        f_tot = f_to_add
    else:
        f_tot = f_tot.merge(f_to_add, how='outer').sort_values('frame')
    
    
# %
diameter = 9  # particle typical diameter Chlamis au x20
tp.quiet()

# search_range =2*diameter #isotropic
search_range =[5.0*diameter,5.0*diameter] #Anisotropic

# search_range =[20.0*diameter,5*diameter] #Anisotropic
pos_columns = ['x','y']#, 'mass']

tracked_traj = tp.link(f_tot, search_range, pos_columns, t_column='frame', memory=20, predictor=None, adaptive_stop=0.5,
                    adaptive_step=0.9, neighbor_strategy=None, link_strategy=None, dist_func=None, to_eucl=None)

tracked_traj.sort_values(['particle', 'frame'], ascending=[
                      True, True], inplace=True)
print('Linking done')

tracked_traj['time'] = tracked_traj['frame'] * 1/fec
tracked_traj['x'] = (tracked_traj['x'])*echelle
tracked_traj['y'] = tracked_traj['y']*echelle

fps=fec

print('saving')
savemat(os.path.join(savefolder,'trajectories_Marc.mat'), tracked_traj)#{'arr':x})#,'x':x_tot_table,'y':y_tot_table,'mass':mass_tot_table})
print('save succesfull')

print('saving')
link_path=os.path.join(savefolder, 'tracked_traj_Marc.csv')
tracked_traj.to_csv(link_path)
print('save succesfull')


trajectoires = {}
for n_particle, traj in tracked_traj.groupby("particle"):
    trajectoires[n_particle] = traj
    
x_tot_table={}
y_tot_table={}
t_tot_table={}
length_table=[]
true_kij=0
for kii in range(0, len(trajectoires), 1):
    if len(trajectoires[kii]["x"].to_numpy())>40*fec: # Si tu veux filtrer par la taille des trajectoires
        if max(trajectoires[kii]["x"].to_numpy())-min(trajectoires[kii]["x"].to_numpy())>0e-5:
            x_tot_table[true_kij]=trajectoires[kii]["x"].to_numpy()
            y_tot_table[true_kij]=trajectoires[kii]["y"].to_numpy()
            t_tot_table[true_kij]=trajectoires[kii]["time"].to_numpy()
            
            # size_tot_table[true_kij]=trajectoires[kii]["size"].to_numpy()
            # ecc_tot_table[true_kij]=trajectoires[kii]["ecc"].to_numpy()
            # signal_tot_table[true_kij]=trajectoires[kii]["signal"].to_numpy()
            # raw_mass_tot_table[true_kij]=trajectoires[kii]["raw_mass"].to_numpy()
            # # ep_tot_table[true_kij]=trajectoires[kii]["ep"].to_numpy()

            length_table.append(len(x_tot_table[true_kij]))
            true_kij=true_kij+1

    

nbr_traj=len(t_tot_table) # total traj number
print(nbr_traj)
longest_index=np.argmax(length_table)
# get the lengths of each trajectory :
legend_for_data=r'$1$ mbar $10$ s'
colors = plt.cm.gnuplot(np.linspace(0,1,nbr_traj))



for kij in range(0, nbr_traj, 1):

    t=t_tot_table[kij]
    x=x_tot_table[kij]
    y=y_tot_table[kij]

    plt.figure(1)

    plt.subplot(211)
    plt.plot(x,y, color=colors[kij])
    plt.xlabel('$x$ $($m$)$')
    plt.ylabel('$y$ $($m$)$')

    plt.subplot(223)
    plt.plot(t,x, color=colors[kij])
    plt.xlabel('$t$ $($s$)$')
    plt.ylabel('$x$ $($m$)$')
    
    plt.subplot(224)
    plt.plot(t,y, color=colors[kij])
    plt.xlabel('$t$ $($s$)$')
    plt.ylabel('$y$ $($m$)$')






# # %% Paramètres
frames = gray(pims.open( os.path.join(target_folder,"*.TIF") ))
print('Number of frames found :')
print(len(frames) )


# %
print('background calculations')
average_list_number=round(4*fec)
average_index_list = range(0,len(frames),average_list_number)
# average_list_number=1
# average_index_list = range(0,50,average_list_number)
image_background = np.median(frames[average_index_list], axis=0)

print("job's done")
# plt.figure(1111111111); 
# plt.subplot(221);
# plt.title('100th image')
# plt.imshow(frames[100])
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# # plt.xlim((100, 300))   # set the xlim to left, right

# plt.subplot(222);
# plt.title('average image')
# plt.imshow(image_background)
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# # plt.xlim((100, 300))   # set the xlim to left, right

# plt.subplot(223);
# plt.title('100th-average')
# plt.imshow((frames[100]-image_background)/image_background)
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# # plt.xlim((100, 300))   # set the xlim to left, right

# plt.subplot(224);
# plt.title('abs((100th-average)/average)')
# plt.imshow(abs(frames[100]-image_background)/image_background)
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

# # %% Un masque pour enlever les bords du canal
# image0=image_background
# gaussian_filter_size_0=5

# blur = cv2.blur(image0, (gaussian_filter_size_0, gaussian_filter_size_0))

# plt.imshow(blur)

# %
    
savefolder3=os.path.join(savefolder,"figs")
if not os.path.exists(savefolder3):
    os.makedirs(savefolder3)

nbr_couleurs=30
# colors_for_particle_list = plt.cm.tab10(np.linspace(0,1,len(all_particules)))
colors_for_particle_list = plt.cm.rainbow(np.linspace(0,1,nbr_couleurs))#len(all_particules)))


# %%

# for frame_index in range(80,120):
#     print(frame_index)
        
#     image000 = frames[frame_index]
#     image00 = (image000-image_background)#/image_background
#     image0 = np.abs(image00)
#     positions_to_plot=tracked_traj[tracked_traj['frame']==frame_index]
#     positions_to_plot['x'] = (positions_to_plot['x'])/echelle
#     positions_to_plot['y'] = positions_to_plot['y']/echelle
    
#     plt.figure(frame_index)
#     plt.subplot(121);
#     plt.imshow(image000)
#     plt.plot([100,100+100e-6/echelle],[10,10],'w')
    
#     for n in positions_to_plot['particle'].unique():
#         to_plot=positions_to_plot[positions_to_plot['particle']==n]
#         plt.plot(to_plot.x,to_plot.y,'o',linewidth=2,markersize=10,fillstyle='none',markeredgewidth=2, markeredgecolor=colors_for_particle_list[n%nbr_couleurs], color=colors_for_particle_list[n%nbr_couleurs])
        
#         to_plot2=tracked_traj[tracked_traj['particle']==n]
#         to_plot2_memory=to_plot2[to_plot2['frame']<frame_index]
#         plt.plot(to_plot2_memory.x/echelle,to_plot2_memory.y/echelle, color=colors_for_particle_list[n%nbr_couleurs])
        
#     plt.subplot(122);
#     plt.imshow(image0)
#     plt.plot([100,100+100e-6/echelle],[10,10],'w')
        
#     for n in positions_to_plot['particle'].unique():
#             to_plot=positions_to_plot[positions_to_plot['particle']==n]
#             plt.plot(to_plot.x,to_plot.y,'o',linewidth=2,markersize=10,fillstyle='none',markeredgewidth=2, markeredgecolor=colors_for_particle_list[n%nbr_couleurs], color=colors_for_particle_list[n%nbr_couleurs])
            
#             to_plot2=tracked_traj[tracked_traj['particle']==n]
#             to_plot2_memory=to_plot2[to_plot2['frame']<frame_index]
#             plt.plot(to_plot2_memory.x/echelle,to_plot2_memory.y/echelle, color=colors_for_particle_list[n%nbr_couleurs])

    
#     time.sleep(1)
#     name_fig=r"im_"+str(frame_index)+r".png"
#     plt.savefig(os.path.join(savefolder3,name_fig), bbox_inches='tight') 
#     time.sleep(1)
#     plt.close(frame_index)

