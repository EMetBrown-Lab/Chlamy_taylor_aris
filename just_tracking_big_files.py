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

#
#
#
#
#

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


# Remains 0.5 to be done
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
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_2s_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_2s_40fps_long\TIF"
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
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_1s_40fps_long\TIF"

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
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__2\TIF"
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__3\TIF"


echelle=180e-6/110 #scale of one pixel
fec = 40. # acquisition frequency 
# echelle=500e-6/1400 #scale of one pixel



savefolder=os.path.join(target_folder,"tracking_results")
if not os.path.exists(savefolder):
    os.makedirs(savefolder)



# # %% Paramètres
frames = gray(pims.open( os.path.join(target_folder,"*.TIF") ))
print('Number of frames found :')
print(len(frames) )




# P_list_manual=np.array([30, 60, 90, 120, 150])
# U_list_manual=echelle*fec*np.array([-50./100, 250./50, 520./25, 864./25, 625./11])
# plt.figure(1111111110); 

# plt.plot(P_list_manual,  U_list_manual,'o',linewidth=2,markersize=10,fillstyle='none',markeredgewidth=2, markeredgecolor=[0, 0, 0])

# plt.plot(P_typical,  echelle*fec*U_typical,'*',linewidth=2,markersize=40,fillstyle='none',markeredgewidth=2, markeredgecolor=[0, 1, 0])

# plt.xlabel(r'$P$ $($mbar$)$')
# plt.ylabel(r'$U$ manual $($m/s$)$')
# plt.title(r'Chip A: $100\,\mu$m $\times 500\,\mu$m $\times 88\,$mm ')

# # plt.title(r"$ $ chanel A : ")

# plt.grid(color='k', linestyle='-', linewidth=0.5)

# # %%

# P_list_1s=[2,2,5,5,10,10]
# dx=np.array([144,240,320,290,200,210])
# dt=np.array([50,60,40,30,35,42])
# v=dx/dt
# v=fec*echelle*v
# plt.figure(1111111111); 

# plt.plot(P_list_1s,  v,'o',linewidth=2,markersize=10,fillstyle='none',markeredgewidth=2, markeredgecolor=[0, 0, 0])

# plt.xlabel(r'$P_0$ $($mbar$)$')
# plt.ylabel(r'$U$ manual $($m/s$)$')
# plt.title(r'Chip C: $100\,\mu$m $\times 180\,\mu$m $\times 88\,$mm ')


# %%
print('background calculations')
average_list_number=round(4*fec)
average_index_list = range(0,len(frames),average_list_number)
# average_list_number=1
# average_index_list = range(0,50,average_list_number)
image_background = np.median(frames[average_index_list], axis=0)

print("job's done")
plt.figure(1111111111); 
plt.subplot(221);
plt.title('100th image')
plt.imshow(frames[100])
plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

plt.subplot(222);
plt.title('average image')
plt.imshow(image_background)
plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

plt.subplot(223);
plt.title('100th-average')
plt.imshow((frames[100]-image_background)/image_background)
plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

plt.subplot(224);
plt.title('abs((100th-average)/average)')
plt.imshow(abs(frames[100]-image_background)/image_background)
plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

# # %% Un masque pour enlever les bords du canal
# image0=image_background
# gaussian_filter_size_0=5

# blur = cv2.blur(image0, (gaussian_filter_size_0, gaussian_filter_size_0))

# plt.imshow(blur)


# %% Margin choices
margin_L=600;
margin_R=10;
margin_T=100;
margin_B=100;

margin_L=0;
margin_R=0;
margin_T=40;
margin_B=80;

margin_L=0;
margin_R=0;
margin_T=0;
margin_B=0;

TRONCATE=True
height=len(image_background)
length=len(image_background[0])
# % Who needs a mask ?

plt.subplot(222);
plt.title('average image')
plt.imshow(image_background)
plt.plot([margin_L,length-margin_R],[margin_T,margin_T],'y')
plt.plot([margin_L,length-margin_R],[height-margin_B,height-margin_B],'y')
plt.plot([margin_L,margin_L],[height-margin_B,margin_T],'y')
plt.plot([length-margin_R,length-margin_R],[height-margin_B,margin_T],'y')

Mask=1*image_background
Mask[:,length-margin_R:length]=0
Mask[:,0:margin_L]=0
Mask[height-margin_B:height,:]=0
Mask[0:margin_T,:]=0
# Mask[Mask==0]=np.nan
plt.imshow(Mask)
# image_background=(Mask)*image_background
# %%






noise_size = 2  # typical size of the noise
smoothing_size = None  # 11 # gaussian filtering for noise
invert = False

topn = 300 # maximum particle number to locate
# might be changed depending on the particles qualities


# # # # Chlamis au x20
diameter = 35  # particle typical diameter Chlamis au x20
maxsize = 20
minsize = 15  # band pass filter to find the relevent feature by size
percentile = 0
separation = 2*diameter  # space between two particles on a single frame
gaussian_filter_size=5
# might be changed depending on the image qualities
minmass_coef=3.0
threshold_coef=minmass_coef






# # # # Chlamy au x5
diameter = 9  # particle typical diameter Chlamis au x20
maxsize = 13
minsize = 5  # band pass filter to find the relevent feature by size
percentile = 0
separation = 2*diameter  # space between two particles on a single frame
gaussian_filter_size=1
# might be changed depending on the image qualities
minmass_coef=1.25
threshold_coef=minmass_coef



spread_for_slipping_average=10
sapce=1
ii = 30

number_processes=19
nombre_image=len(frames)#//4

def locating(i): 
     
    if i<spread_for_slipping_average+1:
        local_average_index_list = range(0,i+2*spread_for_slipping_average,sapce)

    elif i>len(frames)-spread_for_slipping_average-1:
        local_average_index_list = range(i-2*spread_for_slipping_average,nombre_image,sapce)

    else :
        local_average_index_list = range(i-spread_for_slipping_average,i+spread_for_slipping_average,sapce)

    slipping_background = np.median(frames[local_average_index_list], axis=0)
    
    
    image_background_=0.5*(image_background+slipping_background)
    image_background_=slipping_background

    image000 = frames[i][margin_T:height-margin_B,margin_R:length-margin_R]
    image00 = (image000-image_background_[margin_T:height-margin_B,margin_R:length-margin_R])#/image_background
    image0 = np.abs(image00)
        
    blur = cv2.blur(image0, (gaussian_filter_size, gaussian_filter_size))
    
    minmass = minmass_coef*np.std(blur)
    # # kill everything under this background intensity value
    threshold = threshold_coef*np.std(blur)
    # f2 = tp.locate(blur, diameter, minmass, maxsize, separation, noise_size, smoothing_size, threshold, invert, percentile,
    #                topn, preprocess=False, max_iterations=5, filter_before=None, filter_after=None, characterize='false', engine='numba')
    f2 = tp.locate(blur, diameter, minmass, maxsize, separation, noise_size, smoothing_size, threshold, invert, percentile,
                      topn, preprocess=True, max_iterations=5, filter_before=None, filter_after=None, characterize='false', engine='numba')

    
    f2["frame"] = [i] * len(f2) 
    return f2

# # %%





# # %%
tp.quiet()

if ii<spread_for_slipping_average+1:
    local_average_index_list = range(0,ii+2*spread_for_slipping_average,sapce)

elif ii>len(frames)-spread_for_slipping_average-1:
    local_average_index_list = range(ii-2*spread_for_slipping_average,nombre_image,sapce)

else :
    local_average_index_list = range(ii-spread_for_slipping_average,ii+spread_for_slipping_average,sapce)

slipping_background = np.median(frames[local_average_index_list], axis=0)


image_background_=0.5*(image_background+slipping_background)
image000 = frames[ii][margin_T:height-margin_B,margin_R:length-margin_R]
image00 = (image000-image_background_[margin_T:height-margin_B,margin_R:length-margin_R])#/image_background
image0 = np.abs(image00)
blur = cv2.blur(image0, (gaussian_filter_size, gaussian_filter_size))

f_0=locating(ii)

plt.figure(ii);
print(ii)
plt.subplot(221);
tp.annotate(f_0, image000);plt.title('base')
plt.plot([10,10+100e-6/echelle],[10,10],'w')
plt.plot([20,20+diameter],[20,20],'b')

plt.subplot(222);
tp.annotate(f_0, image00);plt.title('base-fond')
plt.plot([10,10+100e-6/echelle],[10,10],'w')
plt.subplot(223);
tp.annotate(f_0, image0);plt.title('abs(base-fond)')
plt.plot([10,10+100e-6/echelle],[10,10],'w')
plt.subplot(224);
tp.annotate(f_0, blur);plt.title('filtered')
plt.plot([10,10+100e-6/echelle],[10,10],'w')
plt.plot([20,20+diameter],[20,20],'b')

# plt.subplot(223);
# plt.imshow(image0);plt.title('filtrage')
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# plt.xlim((100, 300))   # set the xlim to left, right

# plt.subplot(224);
# tp.annotate(f_0, image0);plt.title('detection')
# plt.plot([10,10+100e-6/echelle],[10,10],'w')
# # plt.xlim((100, 300))   # set the xlim to left, right

time.sleep(1)
name_fig=r"reference"+str(ii)+r".png"
plt.savefig(os.path.join(savefolder,name_fig), bbox_inches='tight') 
time.sleep(1)

    
# %%
number_slices=4

slice_size=len(frames)//number_slices
for index_slice in range(number_slices):
    
    first_image=index_slice*slice_size
    nb_image=slice_size
    last_image=(index_slice+1)*slice_size
    
    exec(open(r"C:\Users\m.lagoin\Documents\Python Scripts\parallel.py").read())

