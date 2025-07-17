# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:41:27 2025

@author: m.lagoin
"""

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
from scipy.signal import correlate
from scipy.io import savemat

import trackpy as tp
from pandas import DataFrame, Series  # for convenience
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit as cf
import matplotlib.pyplot as plt
import csv


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

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import buttord

def change_dict_key_exist(d, old_key, new_key):
            if old_key in d:
                d[new_key] = d.pop(old_key)
def switch_x_and_y(d):
        print(d)
        change_dict_key_exist(d, 'x', 'old_x')
        change_dict_key_exist(d, 'y', 'x')
        change_dict_key_exist(d, 'old_x', 'y')
        print(d)
        print('x and y have been permuted')


def compute_average_position_for_all_scales(data_,scales_list_in_points_,fps_):
    position_all_scales_=np.nan*np.zeros(shape = (len(scales_list_in_points_),len(data_)));
    
    for kk in range(len(scales_list_in_points_)):
        # pos_=0.5*(data_[scales_list_[kk]+1:]+data_[1:-scales_list_[kk]]);
        scale_=scales_list_in_points_[kk]
        position_to_average_=np.nan*np.zeros(shape = (len(data_),scale_));
        
        for ii in range(len(data_)-scale_):
            position_to_average_[ii,:]=data_[ii:scale_+ii]
   
        pos_=np.nanmean(position_to_average_,1)
        position_all_scales_[kk,0:len(pos_)]=pos_[:];
        del pos_
    return position_all_scales_

def sin_fun(x,amplitude,frequence,phase,K):
    return K+amplitude*np.sin(2*np.pi*frequence*x+phase)



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


def demodulate_wavelets_siignal_old(t,x_tot,fs,how_many_periods,lag_choice,Frequency_guess,visible):

    size_window=int(how_many_periods*1/Frequency_guess*fs) # cinq periodes c'est raisonnable pour obtenir le bon bruit.
    lag=1#*int(1.5*size_window)
    
    t_in_points=t*fs
    t_in_points=t_in_points.astype(np.int32)

    # ii_max=int((len(x_tot)-size_window)/lag) Pas bon, à vérifier.....
    nbr_possible_windows=int(np.floor((len(x_tot)-size_window)/lag))
    
    
    A_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    K_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    Phase_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    freq_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    
    
    displacement_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    t_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    
    
    for index_window in range(nbr_possible_windows):
        x_tot1=x_tot[index_window*lag:index_window*lag+size_window]
        t1=t_in_points[index_window*lag:index_window*lag+size_window]
        Offset_guess=np.mean(x_tot1)
        Amplitude_guess=1*np.std(x_tot1)
        Phase_guess=1
        
                
        try:
            p_opt1,p_cov=cf(sin_fun,t1,x_tot1, p0=(Amplitude_guess,Frequency_guess/fs,Phase_guess,Offset_guess))
            
            x_retrieved_from_fit1=sin_fun(t1,*p_opt1)
            motion_estimate1=x_tot1-x_retrieved_from_fit1+p_opt1[3]
            
            t1=t1/fs
            displacement_table[index_window,index_window*lag:index_window*lag+size_window]=motion_estimate1
            t_table[index_window,index_window*lag:index_window*lag+size_window]=t1
        
            A_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[0]+0*x_tot1
            K_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[1]+0*x_tot1
            Phase_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[2]+0*x_tot1
            freq_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[3]+0*x_tot1

            del x_tot1, t1, motion_estimate1
        except:
            print("No fit") 
        # else:
            # print('ok, doing it')
        # finally:
        #     print('Anyway, doing that....')
    A_mean_list=np.nanmean(A_table,0)
    K_mean_list=np.nanmean(K_table,0)
    Phase_mean_list=np.nanmean(Phase_table,0)
    freq_mean_list=np.nanmean(freq_table,0)
    
    displacement_mean_list=np.nanmean(displacement_table,0)
    displacement_mean_std=np.nanstd(displacement_table,0)
    
    # displacement_mean_list_filtered=gaussian_filter1d(displacement_mean_list, Frequency_guess*fs)
    displacement_mean_list_filtered=gaussian_filter1d(displacement_mean_list, 0.1*Frequency_guess*fs)

    t_mean_list=np.nanmean(t_table,0)

    if visible==1:
            plt.figure()                                                                                                                                                                                                                                                       
            plt.suptitle(r'fit: $X(t) = K + A $sin$(2\pi f (t+l)+ \phi )$')
            # # plt.subplot(121)
            # plt.xlabel('$t$ $($s$)$')
            # plt.ylabel('$x$ $($m$)$')
            # plt.plot(t,x_tot,linestyle='-',linewidth=5,color='blue',label='total')
    
            # plt.grid(color='k', linestyle='-', linewidth=0.5)
            # plt.subplot(122)
            plt.plot(t,x_tot,linestyle='-',linewidth=5,color='blue',label='total')
            plt.plot(t[0:size_window],x_tot[0:size_window],linestyle='-',linewidth=3,color='orange',label='window')
            plt.plot(t_mean_list,displacement_mean_list,linestyle='-',linewidth=4,color='red',label='wavelett')
            plt.plot(t_mean_list,displacement_mean_list+2*displacement_mean_std,linestyle='--',linewidth=0.5,color='red',label='')#,color='yellow',label='total-fit1')
            plt.plot(t_mean_list,displacement_mean_list-2*displacement_mean_std,linestyle='--',linewidth=0.5,color='red',label='')#,color='yellow',label='total-fit1')
            plt.plot(t_mean_list,displacement_mean_list_filtered,linestyle='-',linewidth=2,color='lime',label='filtered')

            plt.xlabel('$t$ $($s$)$')
            plt.ylabel('$x_b$ $($m$)$')
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend()
            plt.tight_layout()

    return(t_mean_list,displacement_mean_list_filtered,displacement_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list)
               

def demodulate_wavelets_siignal(t,x_tot,fs,how_many_periods,lag_choice,Frequency_guess,visible):

    size_window=int(how_many_periods*1/Frequency_guess*fs) # cinq periodes c'est raisonnable pour obtenir le bon bruit.
    lag=1#*int(1.5*size_window)
    
    t_in_points=t*fs
    t_in_points=t_in_points.astype(np.int32)

    # ii_max=int((len(x_tot)-size_window)/lag) Pas bon, à vérifier.....
    nbr_possible_windows=int(np.floor((len(x_tot)-size_window)/lag))
    
    
    A_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    K_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    Phase_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    freq_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    
    
    displacement_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    t_table=np.nan*np.zeros(shape = (nbr_possible_windows,len(x_tot)));
    
    
    for index_window in range(nbr_possible_windows):
        x_tot1=x_tot[index_window*lag:index_window*lag+size_window]
        t1=t_in_points[index_window*lag:index_window*lag+size_window]
        Offset_guess=np.mean(x_tot1)
        Amplitude_guess=1*np.std(x_tot1)
        Phase_guess=1
        
                
        try:
            p_opt1,p_cov=cf(sin_fun,t1,x_tot1, p0=(Amplitude_guess,Frequency_guess/fs,Phase_guess,Offset_guess))
            
            x_retrieved_from_fit1=sin_fun(t1,*p_opt1)
            motion_estimate1=x_tot1-x_retrieved_from_fit1+p_opt1[3]
            
            t1=t1/fs
            displacement_table[index_window,index_window*lag:index_window*lag+size_window]=motion_estimate1
            t_table[index_window,index_window*lag:index_window*lag+size_window]=t1
        
            A_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[0]+0*x_tot1
            K_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[1]+0*x_tot1
            Phase_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[2]+0*x_tot1
            freq_table[index_window,index_window*lag:index_window*lag+size_window]=p_opt1[3]+0*x_tot1

            del x_tot1, t1, motion_estimate1
        except:
            print("No fit") 
        # else:
            # print('ok, doing it')
        # finally:
        #     print('Anyway, doing that....')
    A_mean_list=np.nanmean(A_table,0)
    K_mean_list=np.nanmean(K_table,0)
    Phase_mean_list=np.nanmean(Phase_table,0)
    freq_mean_list=np.nanmean(freq_table,0)
    
    displacement_mean_list=np.nanmean(displacement_table,0)
    displacement_mean_std=np.nanstd(displacement_table,0)
    
    # displacement_mean_list_filtered=gaussian_filter1d(displacement_mean_list, Frequency_guess*fs)
    displacement_mean_list_filtered=gaussian_filter1d(displacement_mean_list, 0.1*Frequency_guess*fs)

    t_mean_list=np.nanmean(t_table,0)

    if visible==1:
            plt.figure()                                                                                                                                                                                                                                                       
            plt.suptitle(r'fit: $X(t) = K + A $sin$(2\pi f (t+l)+ \phi )$')
            # # plt.subplot(121)
            # plt.xlabel('$t$ $($s$)$')
            # plt.ylabel('$x$ $($m$)$')
            # plt.plot(t,x_tot,linestyle='-',linewidth=5,color='blue',label='total')
    
            # plt.grid(color='k', linestyle='-', linewidth=0.5)
            # plt.subplot(122)
            plt.plot(t,x_tot,linestyle='-',linewidth=5,color='blue',label='total')
            plt.plot(t[0:size_window],x_tot[0:size_window],linestyle='-',linewidth=3,color='orange',label='window')
            plt.plot(t_mean_list,displacement_mean_list,linestyle='-',linewidth=4,color='red',label='wavelett')
            plt.plot(t_mean_list,displacement_mean_list+2*displacement_mean_std,linestyle='--',linewidth=0.5,color='red',label='')#,color='yellow',label='total-fit1')
            plt.plot(t_mean_list,displacement_mean_list-2*displacement_mean_std,linestyle='--',linewidth=0.5,color='red',label='')#,color='yellow',label='total-fit1')
            plt.plot(t_mean_list,displacement_mean_list_filtered,linestyle='-',linewidth=2,color='lime',label='filtered')

            plt.xlabel('$t$ $($s$)$')
            plt.ylabel('$x_b$ $($m$)$')
            plt.grid(color='k', linestyle='-', linewidth=0.5)
            plt.legend()
            plt.tight_layout()

    return(t_mean_list,displacement_mean_list_filtered,displacement_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list)
               


def histogram_norm_marc(a,nbins):
    # note que np.sum(hist * np.diff(bin_edges))=1.0
    # nbins=len(a)//100
    a2=a[~np.isnan(a)]
    hist, bin_edges = np.histogram(a2,nbins, density=True)
    min_edge=min(bin_edges)
    lapse=np.mean(np.diff(bin_edges))
    hist_x=[min_edge+lapse/2+n*lapse for n in range(len(hist))]
    hist_y=hist
    return np.array(hist_x),np.array(hist_y)

def undersample_Marc(t,x,y,new_period,fs,visible):
    """
    t,x,y,
    new_period en secondes
    fs original sampling frequency
    visible=1 if you want plots
    """
    period=round(new_period*fs)
    new_fs=1/new_period
    length_=round(len(x)/new_period/fs)
    x_ech_=np.nan*np.zeros(shape = (period,length_));
    y_ech_=np.nan*np.zeros(shape = (period,length_));
    t_ech_=np.nan*np.zeros(shape = (period,length_));
    
    colors = plt.cm.summer(np.linspace(0,1,period))
    for tt in range(0,period):
        if len(range(tt,len(x),period))==length_:
            # print(tt)
            x_ech_[tt,:]=x[range(tt,len(x),period)]
            y_ech_[tt,:]=y[range(tt,len(x),period)]
            t_ech_[tt,:]=t[range(tt,len(y),period)]   
        if len(range(tt,len(x),period))==length_-1:
            x_ech_[tt,:-1]=x[range(tt,len(x),period)]
            y_ech_[tt,:-1]=y[range(tt,len(x),period)]
            t_ech_[tt,:-1]=t[range(tt,len(y),period)]
            
    if visible==1:
        plt.figure()
        plt.subplot(211)
        plt.plot(x,y,linestyle='-',linewidth=2,color='red',label='wavelett')
        plt.xlabel(r'$x$ $($m$)$')
        plt.ylabel(r'$y$ $($m$)$')
        plt.grid(color='k', linestyle='-', linewidth=0.5)
        plt.subplot(223)
        plt.plot(t,x,linestyle='-',linewidth=2,color='red',label='wavelett')
        plt.ylabel(r'$x$ $($m$)$')
        plt.xlabel(r'$t$ $($s$)$')
        plt.grid(color='k', linestyle='-', linewidth=0.5)
        plt.subplot(224)
        plt.plot(t,y,linestyle='-',linewidth=2,color='red',label='wavelett')
        plt.ylabel(r'$y$ $($m$)$')
        plt.xlabel(r'$t$ $($s$)$')
        plt.grid(color='k', linestyle='-', linewidth=0.5)
        for tt in range(0,period-1,20):
            plt.subplot(211)
            plt.plot(x_ech_[tt,:],y_ech_[tt,:], color=colors[tt],alpha=0.75)
            
            plt.subplot(223)
            plt.plot(t_ech_[tt,:],x_ech_[tt,:], color=colors[tt],alpha=0.75)
            
            plt.subplot(224)
            plt.plot(t_ech_[tt,:],y_ech_[tt,:], color=colors[tt],alpha=0.75)
        
    return t_ech_,x_ech_,y_ech_,new_fs



V0_set_average_list=[]
V0_set_average_err_list=[]
D_set_average_list=[]
D_set_average_err_list=[]
D_alpha_set_average_list=[]
D_alpha_set_average_err_list=[]

A_list=[]
T_list=[]


TA_estimate_list_fit=[]
TA_estimate_list_fit_err=[]
Tau_x_estimate_list=[]
Tau_y_estimate_list=[]
Tau_x_estimate_err_list=[]
Tau_y_estimate_err_list=[]
TA_estimate_list=[]
TA_estimate_list_err=[]


V0x_set_average_list=[]
V0x_set_average_err_list=[]
V0y_set_average_list=[]
V0y_set_average_err_list=[]

V0x_fit_set_average_fit_list=[]
V0y_fit_set_average_fit_list=[]
Dx_fit_set_average_fit_list=[]
Dy_fit_set_average_fit_list=[]
Taux_fit_estimate_list=[]
Tauy_fit_estimate_list=[]
TA_fit_estimate_list=[]
V0x_fit_set_average_fit_list_err=[]
V0y_fit_set_average_fit_list_err=[]
Dx_fit_set_average_fit_list_err=[]
Dy_fit_set_average_fit_list_err=[]
Taux_fit_estimate_list_err=[]
Tauy_fit_estimate_list_err=[]
TA_fit_estimate_list_err=[]


D_ech_set_average_list=[]
D_ech_set_average_err_list=[]

# visible=1
# t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[longest_index],x_tot_table[longest_index],fs,how_many_periods,lag_choice,Frequency_guess,visible)
save=0
# % Curves for the article

just_for_show=0


colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,9))
Ratio=1
target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
color_for_data=[0.5,0.5,0.5]
Amp=0
Period=2
legend_for_data=r'$0$ mbar $2$ s'
Frequency_guess=4
how_many_periods=3






# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long_after\TIF"
# color_for_data=[0,1,0]
# Amp=0
# Period=2
# legend_for_data=r'$0$ mbar $2$ afters'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=1
# legend_for_data=r'$0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long_apres\TIF"
# color_for_data=[0,0.3,0.3]
# Amp=0
# Period=1
# legend_for_data=r'$0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
# color_for_data=[0.1,0,0]
# Amp=0
# Period=0.5
# legend_for_data=r'$0$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long_apres\TIF"
# color_for_data=[0.1,0.1,0.1]
# Amp=0
# Period=0.5
# legend_for_data=r'$0$ mbar $0.5$ afters'
# Frequency_guess=2
# how_many_periods=3



# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# color_for_data=[0.5,0.5,0.5]
# Amp=0
# Period=2
# legend_for_data=r'C $0$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# # # # # # # # mon pref'
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*1]
# Amp=1
# Period=2
# legend_for_data=r'$1$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3
# # # # # # # # # # # # # # # # # # # mon pref'
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*2]
# Amp=2
# Period=2
# legend_for_data=r'$2$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*4]
# Amp=4
# Period=2
# legend_for_data=r'$4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*8]
# Amp=8
# Period=2
# legend_for_data=r'$8$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3



# colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,13))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=2
# legend_for_data=r'C $0$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=2
# legend_for_data=r'C $1$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=2
# legend_for_data=r'C $2$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=2
# legend_for_data=r'C $3$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=2
# legend_for_data=r'C $4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*6)]
# Amp=6
# Period=2
# legend_for_data=r'C $6$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*8)]
# Amp=8
# Period=2
# legend_for_data=r'C $8$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*12)]
# Amp=12
# Period=2
# legend_for_data=r'C $12$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*16)]
# # # # # # # Amp=16
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $16$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*20)]
# # # # # # # Amp=20
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $20$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_24mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*24)]
# # # # # # # Amp=24
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $24$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_28mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*28)]
# # # # # # # Amp=28
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $28$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*32)]
# # # # # # # Amp=32
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $32$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3























# # # Mon pref' for figures
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=2
# legend_for_data=r'C $4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3
# just_for_show=1


# # # # %



# colors_for_pressures_list = plt.cm.winter(np.linspace(0,1,16))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
# color_for_data=[0,0,0.1]
# Amp=0
# Period=2
# legend_for_data=r'$0$ mbar $2$ s'
# Frequency_guess=4
# how_many_periods=3


# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*1]
# Amp=1
# Period=4
# legend_for_data=r'$1$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*2]
# Amp=2
# Period=4
# legend_for_data=r'$2$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*4]
# Amp=4
# Period=4
# legend_for_data=r'$4$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3
# # # # save=1



# colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,16))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long_after\TIF"
# color_for_data=[0,0.1,0]
# Amp=0
# Period=2
# legend_for_data=r'$0$ mbar $2$ afters'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*1]
# Amp=1
# Period=2
# legend_for_data=r'$1$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*2]
# Amp=2
# Period=2
# legend_for_data=r'$2$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*4]
# Amp=4
# Period=2
# legend_for_data=r'$4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*8]
# Amp=8
# Period=2
# legend_for_data=r'$8$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3
# # save=1



# colors_for_pressures_list = plt.cm.spring(np.linspace(0,1,35))
# Ratio=1
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=1
# legend_for_data=r'$0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long_apres\TIF"
# color_for_data=[0,0.3,0.3]
# Amp=0
# Period=1
# legend_for_data=r'$0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# # # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# # # # # # color_for_data=[0,0.5,0.5]
# # # # # # # # # Amp=0
# # # # # # # # # Period=2
# # # # # # # # legend_for_data=r'$0$ mbar $1$ s'
# # # # # # # # Frequency_guess=1
# # # # # # # # how_many_periods=3 # Dans celui là, il y a des oscillations....

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*1]
# Amp=1
# Period=1
# legend_for_data=r'$1$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*2]
# Amp=2
# Period=1
# legend_for_data=r'$2$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*4]
# Amp=4
# Period=1
# legend_for_data=r'$4$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*8]
# Amp=8
# Period=1
# legend_for_data=r'$8$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*16]
# Amp=16
# Period=1
# legend_for_data=r'$16$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_1s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[Ratio*32]
# # # # # # Amp=32
# # # # # # Period=1
# # # # # # legend_for_data=r'$32$ mbar $1$ s'
# # # # # # Frequency_guess=1
# # # # # # how_many_periods=3
# # # # # save=1

# colors_for_pressures_list = plt.cm.autumn(np.linspace(0,1,16))
# Ratio=1
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
# color_for_data=[0.1,0,0]
# Amp=0
# Period=0.5
# legend_for_data=r'$0$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long_apres\TIF"
# color_for_data=[0.3,0,0]
# Amp=0
# Period=0.5
# legend_for_data=r'$0$ mbar $0.5$ afters'
# Frequency_guess=2
# how_many_periods=3

# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
# # color_for_data=colors_for_pressures_list[Ratio*1]
# # Amp=1
# # Period=0.5
# # legend_for_data=r'$1$ mbar $0.5$ s'
# # Frequency_guess=2
# # how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*2]
# Amp=2
# Period=0.5
# legend_for_data=r'$2$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[Ratio*4]
# Amp=4
# Period=0.5
# legend_for_data=r'$4$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_0p5s_40fps_long\TIF"
# # color_for_data=colors_for_pressures_list[Ratio*8]
# # Amp=8
# # Period=0.5
# # legend_for_data=r'$8$ mbar $0.5$ s'
# # Frequency_guess=2
# # how_many_periods=3
# # save=1

# # # # # # celui la ne marche pas
# # # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\0p5\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_0p5s_40fps_long\TIF"
# # # # # # # # color_for_data=colors_for_pressures_list[Ratio*16]
# # # # # # # # legend_for_data=r'$16$ mbar $1$ s'
# # # # # # # # Frequency_guess=1
# # # # # # # # how_many_periods=3






# # # %
# # # Puce A
# colors_for_pressures_list = plt.cm.winter(np.linspace(0,1,17))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s"
# Ratio=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
# color_for_data=[0,0,0]
# Amp=0
# Period=4
# legend_for_data=r'A $0$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long_apres\TIF"
# color_for_data=[0,0,0]
# Amp=0
# Period=4
# legend_for_data=r'A $0$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[Ratio*1]
# # # # # # Amp=1
# # # # # # Period=4
# # # # # # legend_for_data=r'A $1$ mbar $4$ s'
# # # # # Frequency_guess=0.25
# # # # # # how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1.33)]
# Amp=1.33
# Period=4
# legend_for_data=r'A $1.33$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=4
# legend_for_data=r'A $2$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=2.66
# Period=4
# legend_for_data=r'A $2.66$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=4
# legend_for_data=r'A $3$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=4
# legend_for_data=r'A $4$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5)]
# Amp=5
# Period=5
# legend_for_data=r'A $5$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\4s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5.33)]
# Amp=5.33
# Period=4
# legend_for_data=r'A $5.33$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3
# # # save=1

# colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,17))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s"
# Ratio=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# color_for_data=[0,0.1,0]
# Amp=0
# Period=2
# legend_for_data=r'A $0$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long_apres\TIF"
# color_for_data=[0,0.1,0]
# Amp=0
# Period=2
# legend_for_data=r'A $0$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=2
# legend_for_data=r'A $1$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1.33)]
# Amp=1.33
# Period=2
# legend_for_data=r'A $1.33$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=2
# legend_for_data=r'A $2$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2.66)]
# Amp=2.66
# Period=2
# legend_for_data=r'A $2.66$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=2
# legend_for_data=r'A $3$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=2
# legend_for_data=r'A $4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5)]
# Amp=5
# Period=2
# legend_for_data=r'A $5$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\2s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5.33)]
# Amp=5.33
# Period=2
# legend_for_data=r'A $5.33$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3
# # save=1


# colors_for_pressures_list = plt.cm.spring(np.linspace(0,1,35))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s"
# Ratio=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=1
# legend_for_data=r'A $0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long_after\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=1
# legend_for_data=r'A $0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=1
# legend_for_data=r'A $1$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1.33)]
# Amp=1.33
# Period=1
# legend_for_data=r'A $1.33$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=1
# legend_for_data=r'A $2$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2.66)]
# Amp=2.66
# Period=1
# legend_for_data=r'A $2.66$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=1
# legend_for_data=r'A $3$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=1
# legend_for_data=r'A $4$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5)]
# Amp=5
# Period=1
# legend_for_data=r'A $5$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*5.33)]
# Amp=5.33
# Period=1
# legend_for_data=r'A $5.33$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_6mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*6)]
# Amp=6
# Period=1
# legend_for_data=r'A $6$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\1s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_10mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*10)]
# Amp=10
# Period=1
# legend_for_data=r'A $10$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3
# save=1


# colors_for_pressures_list = plt.cm.autumn(np.linspace(0,1,17))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s"
# Ratio=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
# color_for_data=[0,0,0.1]
# Amp=0
# Period=0.5
# legend_for_data=r'A $0$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=0.5
# legend_for_data=r'A $1$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_1p33mbar_0p5s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1.33)]
# Amp=1.33
# Period=0.5
# legend_for_data=r'A $1.33$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=0.5
# legend_for_data=r'A $2$ mbar $0.5$ s'
# Frequency_guess=2
# how_many_periods=3

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_2p66mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*2.66)]
# # # Amp=2.66
# # # Period=0.5
# # # legend_for_data=r'A $2.66$ mbar $0.5$ s'
# # # Frequency_guess=2
# # # how_many_periods=3

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_3mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*3)]
# # # Amp=3
# # # Period=0.5
# # # legend_for_data=r'A $3$ mbar $0.5$ s'
# # # Frequency_guess=2
# # # how_many_periods=3

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*4)]
# # # Amp=4
# # # Period=0.5
# # # legend_for_data=r'A $4$ mbar $0.5$ s'
# # # Frequency_guess=2
# # # how_many_periods=3

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*5)]
# # # Amp=5
# # # Period=0.5
# # # legend_for_data=r'A $5$ mbar $0.5$ s'
# # # Frequency_guess=2
# # # how_many_periods=3

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*5.33)]
# # # Amp=5.33
# # # Period=0.5
# # # legend_for_data=r'A $5.33$ mbar $0.5$ s'
# # # Frequency_guess=2
# # # how_many_periods=3


# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_0p5s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*6)]
# # # # # # Amp=6
# # # # # # Period=0.5
# # # # # # legend_for_data=r'A $6$ mbar $0.5$ s'
# # # # # # Frequency_guess=2
# # # # # # how_many_periods=2.5

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\0p5s\Chalmy_in_chanel_A_tamer_filtre_Benjamin_x5_sinei_5p33mbar_0p5s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*10)]
# # # # # # Amp=10
# # # # # # Period=0.5
# # # # # # legend_for_data=r'A $10$ mbar $0.5$ s'
# # # # # # Frequency_guess=2
# # # # # # how_many_periods=2.5
# # # # # # save=1

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*0)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*1)]
# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*2)]
# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*3)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*4)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*5)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*6)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*7)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*8)]
# # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\0p5s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_0p5s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[round(Ratio*9)]






# # # # # %


# colors_for_pressures_list = plt.cm.cool(np.linspace(0,1,5))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_8s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=8
# legend_for_data=r'C $0$ mbar $8$ s'
# Frequency_guess=0.125
# how_many_periods=2

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_8s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=8
# legend_for_data=r'C $1$ mbar $8$ s'
# Frequency_guess=0.125
# how_many_periods=2

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_8s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*2)]
# # # # # # # Amp=2
# # # # # # # Period=8
# # # # # # # legend_for_data=r'C $2$ mbar $4$ s'
# # # # # # # Frequency_guess=0.125
# # # # # # # how_many_periods=2
# # # # # f

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_8s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=8
# legend_for_data=r'C $3$ mbar $8$ s'
# Frequency_guess=0.125
# how_many_periods=2

# # # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\8s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_8s_40fps_long\TIF"
# # # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*4)]
# # # # # # # # Amp=4
# # # # # # # # Period=8
# # # # # # # # legend_for_data=r'C $4$ mbar $4$ s'
# # # # # # # # Frequency_guess=0.125
# # # # # # # # how_many_periods=2


# colors_for_pressures_list = plt.cm.winter(np.linspace(0,1,9))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_4s_40fps_long\TIF"
# color_for_data=[0.1,0,0,0]
# Amp=0
# Period=4
# legend_for_data=r'C $0$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=4
# legend_for_data=r'C $1$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=4
# legend_for_data=r'C $2$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=4
# legend_for_data=r'C $3$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=4
# legend_for_data=r'C $4$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*6)]
# Amp=6
# Period=4
# legend_for_data=r'C $6$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_4s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*8)]
# Amp=8
# Period=4
# legend_for_data=r'C $8$ mbar $4$ s'
# Frequency_guess=0.25
# how_many_periods=2.5

# # # # # pas assez de longue
# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\4s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_4s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*12)]
# # # # # # # Amp=12
# # # # # # # Period=4
# # # # # # # legend_for_data=r'C $12$ mbar $4$ s'
# # # # # # # Frequency_guess=0.25
# # # # # # # how_many_periods=2.5




# colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,13))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_2s_40fps_long\TIF"
# color_for_data=[0.1,0,0.1]
# Amp=0
# Period=2
# legend_for_data=r'C $0$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*1)]
# Amp=1
# Period=2
# legend_for_data=r'C $1$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*2)]
# Amp=2
# Period=2
# legend_for_data=r'C $2$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=2
# legend_for_data=r'C $3$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=2
# legend_for_data=r'C $4$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*6)]
# Amp=6
# Period=2
# legend_for_data=r'C $6$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*8)]
# Amp=8
# Period=2
# legend_for_data=r'C $8$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_2s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*12)]
# Amp=12
# Period=2
# legend_for_data=r'C $12$ mbar $2$ s'
# Frequency_guess=0.5
# how_many_periods=3


# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*16)]
# # # # # # # Amp=16
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $16$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*20)]
# # # # # # # Amp=20
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $20$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_24mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*24)]
# # # # # # # Amp=24
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $24$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_28mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*28)]
# # # # # # # Amp=28
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $28$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3

# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_2s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*32)]
# # # # # # # Amp=32
# # # # # # # Period=2
# # # # # # # legend_for_data=r'C $32$ mbar $2$ s'
# # # # # # # Frequency_guess=0.5
# # # # # # # how_many_periods=3



# colors_for_pressures_list = plt.cm.spring(np.linspace(0,1,33))
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s"
# Ratio=1

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_0mbar_1s_40fps_long\TIF"
# color_for_data=[0.1,0,1]
# Amp=0
# Period=1
# legend_for_data=r'C $0$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_1s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*1)]
# # # # # # Amp=1
# # # # # # Period=1
# # # # # # legend_for_data=r'C $1$ mbar $1$ s'
# # # # # # Frequency_guess=1
# # # # # # how_many_periods=3

# # # # pas bon, il y a eu un souci sur cette exp
# # # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_2mbar_1s_40fps_long\TIF"
# # # # # # # color_for_data=colors_for_pressures_list[round(Ratio*2)]
# # # # # # # Amp=2
# # # # # # # Period=1
# # # # # # # legend_for_data=r'C $2$ mbar $1$ s'
# # # # # # # Frequency_guess=1
# # # # # # # how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_3mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*3)]
# Amp=3
# Period=1
# legend_for_data=r'C $3$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3
# 11 1 1 1 1 1 1 1 1 1%
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_4mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*4)]
# Amp=4
# Period=1
# legend_for_data=r'C $4$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_6mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*6)]
# Amp=6
# Period=1
# legend_for_data=r'C $6$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_8mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*8)]
# Amp=8
# Period=1
# legend_for_data=r'C $8$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3
# 11 1 1 1 1 1 1 1 1 1%
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_12mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*12)]
# Amp=12
# Period=1
# legend_for_data=r'C $12$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_16mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*16)]
# Amp=16
# Period=1
# legend_for_data=r'C $16$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_20mbar_1s_40fps_long\TIF"
# color_for_data=colors_for_pressures_list[round(Ratio*20)]
# Amp=20
# Period=1
# legend_for_data=r'C $20$ mbar $1$ s'
# Frequency_guess=1
# how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_24mbar_1s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*24)]
# # # # # # Amp=24
# # # # # # Period=1
# # # # # # legend_for_data=r'C $24$ mbar $1$ s'
# # # # # # Frequency_guess=1
# # # # # # how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_28mbar_1s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*28)]
# # # # # # Amp=28
# # # # # # Period=1
# # # # # # legend_for_data=r'C $28$ mbar $1$ s'
# # # # # # Frequency_guess=1
# # # # # # how_many_periods=3

# # # # # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\1s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_32mbar_1s_40fps_long\TIF"
# # # # # # color_for_data=colors_for_pressures_list[round(Ratio*32)]
# # # # # # Amp=32
# # # # # # Period=1
# # # # # # legend_for_data=r'C $32$ mbar $1$ s'
# # # # # # Frequency_guess=1
# # # # # # how_many_periods=3
# # # # # f
# # # # %



# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DA1TA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_1\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_2\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\1\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long_3\TIF"

# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2_\TIF"
# target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\12_06_2025\2\Chalmy_in_goute_tamer_filtre_Benjamin_x5_40fps_long2__1\TIF"



if Ratio==3:
    width=500e-6
if Ratio==1:
    width=180e-6

echelle=180e-6/110 #scale of one pixel
fec = 40. # acquisition frequency 
# how_many_periods=5
fs=fec
fps=fec

savefolder=os.path.join(target_folder,"tracking_results")
link_path=os.path.join(savefolder, 'tracked_traj_Marc.csv')
tracked_traj= pd.read_csv(link_path)


trajectoires = {}
for n_particle, traj in tracked_traj.groupby("particle"):
    trajectoires[n_particle] = traj
    
x_tot_table={}
y_tot_table={}
t_tot_table={}
mass_tot_table={}
length_table=[]

true_kij=0
for kii in range(0, len(trajectoires)-1, 1):
    # if len(trajectoires[kii]["x"].to_numpy())>1.5*how_many_periods/Frequency_guess*fec: # Si tu veux filtrer par la taille des trajectoires
        # if max(trajectoires[kii]["x"].to_numpy())-min(trajectoires[kii]["x"].to_numpy())>10e-5:
    # if len(trajectoires[kii]["x"].to_numpy())>40*0.3*how_many_periods/Frequency_guess*fec: # Si tu veux filtrer par la taille des trajectoires
    if len(trajectoires[kii]["x"].to_numpy())>40*fec:#3*how_many_periods*Period*fec: # Si tu veux filtrer par la taille des trajectoires
        # if max(trajectoires[kii]["x"].to_numpy())-min(trajectoires[kii]["x"].to_numpy())>100e-6:
        if Ratio==3:
            if np.nanstd(trajectoires[kii]["y"].to_numpy())>1e-5:
                
                x_tot_table[true_kij]=trajectoires[kii]["x"].to_numpy()
                y_tot_table[true_kij]=trajectoires[kii]["y"].to_numpy()
                t_tot_table[true_kij]=trajectoires[kii]["time"].to_numpy()
                length_table.append(len(x_tot_table[true_kij]))
                true_kij=true_kij+1
        if Ratio==1:
          if np.nanstd(trajectoires[kii]["y"].to_numpy())>3e-5:
            x_tot_table[true_kij]=trajectoires[kii]["x"].to_numpy()
            y_tot_table[true_kij]=trajectoires[kii]["y"].to_numpy()
            t_tot_table[true_kij]=trajectoires[kii]["time"].to_numpy()
            length_table.append(len(x_tot_table[true_kij]))
            true_kij=true_kij+1

nbr_traj=len(t_tot_table) # total traj number
print(nbr_traj)
longest_index=np.argmax(length_table)
longest_index0=np.argmax(length_table)

# get the lengths of each trajectory :

    


fs=fec
lag_choice=1
visible=0
nbins=100
# t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[longest_index],x_tot_table[longest_index],fs,how_many_periods,lag_choice,Frequency_guess,visible)
visible=0
# %

# %
colors = plt.cm.gnuplot(np.linspace(0,1,nbr_traj))
plt.figure(1)
for kij in range(0, nbr_traj, 1):
    # print(np.nanstd(y_tot_table[kij]))
    # t=t_tot_table[kij]
    # x=x_tot_table[kij]
    # y=y_tot_table[kij]
    # t=t-t[0]
    # x=x#-x[0]
    # y=y#-y[0]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(x_tot_table[kij],y_tot_table[kij], color=colors[kij],alpha=0.1)

    plt.subplot(223)
    plt.plot(t_tot_table[kij],x_tot_table[kij], color=colors[kij],alpha=0.1)

    plt.subplot(224)
    plt.plot(t_tot_table[kij],y_tot_table[kij], color=colors[kij],alpha=0.1)
    
plt.figure(1)
plt.subplot(211)
plt.xlabel(r'$x$ $($m$)$')
plt.ylabel(r'$y$ $($m$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(223)
plt.ylabel(r'$x$ $($m$)$')
plt.xlabel(r'$t$ $($s$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(224)
plt.ylabel(r'$y$ $($m$)$')
plt.xlabel(r'$t$ $($s$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)


plt.figure(100)
plt.subplot(211)
plt.plot(x_tot_table[longest_index],y_tot_table[longest_index], color=color_for_data,alpha=0.75)
plt.xlabel(r'$x$ $($m$)$')
plt.ylabel(r'$y$ $($m$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(223)
plt.plot(t_tot_table[longest_index],x_tot_table[longest_index], color=color_for_data,alpha=0.75)
plt.ylabel(r'$x$ $($m$)$')
plt.xlabel(r'$t$ $($s$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(224)
plt.plot(t_tot_table[longest_index],y_tot_table[longest_index], color=color_for_data,alpha=0.75)
plt.ylabel(r'$y$ $($m$)$')
plt.xlabel(r'$t$ $($s$)$')
plt.grid(color='k', linestyle='-', linewidth=0.5)


t_hat_data,x_hat_data,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[longest_index],x_tot_table[longest_index],fs,how_many_periods,lag_choice,Frequency_guess,visible)
x_data=x_tot_table[longest_index]
y_data=y_tot_table[longest_index]
t_data=t_tot_table[longest_index]

alpha_hat=np.nan*np.zeros(shape = (len(x_hat_data)));
alpha0=np.angle(np.diff(x_hat_data)+ 1j*np.diff(y_data) )
alpha_hat[0:len(alpha0)]=alpha0

# Save to CSV
# with open(os.path.join(target_folder,'figure_1_1.csv'), 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['t','x', 'y','t_hat','x_hat','phi_hat'])  # header
#     writer.writerows(zip(t_data, x_data, y_data,t_hat_data,x_hat_data,alpha_hat))

# # %%

# % Calcul des quantités
# %
# % initialisation
data0=x_tot_table[longest_index]; # in case there are two
scales_list0, increments_all_scales0 , scales_list_in_points0=compute_increments_for_all_scales(data0,fps)
Q_2_average0=np.nanvar(increments_all_scales0 ,1 )

number_scales=len(scales_list0)
number_traj=len(x_tot_table)
L_inc=len(increments_all_scales0[0])

scales_list_global_x=np.nan*np.zeros(shape = (number_traj,number_scales));
increments_global_x=np.nan*np.zeros(shape = (number_traj,number_scales,L_inc));
scales_list_global_y=np.nan*np.zeros(shape = (number_traj,number_scales));
increments_global_y=np.nan*np.zeros(shape = (number_traj,number_scales,L_inc));
scales_list_global_alpha=np.nan*np.zeros(shape = (number_traj,number_scales));
increments_global_alpha=np.nan*np.zeros(shape = (number_traj,number_scales,L_inc));
increments_global_x0=np.nan*np.zeros(shape = (number_traj,number_scales,L_inc));

Q2_list_global_x=np.nan*np.zeros(shape = (number_traj,number_scales));
Q2_list_global_y=np.nan*np.zeros(shape = (number_traj,number_scales));
Q2_list_global_alpha=np.nan*np.zeros(shape = (number_traj,number_scales));
Q2_list_global_x0=np.nan*np.zeros(shape = (number_traj,number_scales));



fs=fec
lag_choice=1
visible=0
nbins=50
# %

for kk in range(number_traj):
    # Sur x
    
    scales_listx, incrementsx0, scales_list_in_points=compute_increments_for_all_scales(x_tot_table[kk],fps);
    this_inc_length=len(incrementsx0[0])
    this_number_scales=len(scales_listx)
    # scales_list_global_x[kk,:this_number_scales]=scales_listx;
    increments_global_x0[kk,:this_number_scales,:this_inc_length]=incrementsx0;
    Q2_list_global_x0[kk,:this_number_scales]=np.nanvar(incrementsx0,1);
    
    if kk==longest_index:
        visible=1
    if Amp>0:
        t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[kk],x_tot_table[kk],fs,how_many_periods,lag_choice,Frequency_guess,visible)
    # t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal2(t_tot_table[kk],x_tot_table[kk],fs,how_many_periods,lag_choice,Frequency_guess,visible)
    else:
        t_mean_list=t_tot_table[kk]
        x_hat=x_tot_table[kk]
        
    if kk==longest_index:
        visible=0

    
    scales_listx, incrementsx, scales_list_in_points=compute_increments_for_all_scales(x_hat,fps);
    this_inc_length=len(incrementsx[0])
    this_number_scales=len(scales_listx)
    scales_list_global_x[kk,:this_number_scales]=scales_listx;
    increments_global_x[kk,:this_number_scales,:this_inc_length]=incrementsx;
    Q2_list_global_x[kk,:this_number_scales]=np.nanvar(incrementsx,1);
    
    # del scales_listy, incrementsy, scales_list_in_points
    
    # Sur y
    y_hat=y_tot_table[kk]


    scales_listy, incrementsy, scales_list_in_points=compute_increments_for_all_scales(y_hat,fps);
    scales_list_global_y[kk,:this_number_scales]=scales_listy;
    increments_global_y[kk,:this_number_scales,:this_inc_length]=incrementsy;
    Q2_list_global_y[kk,:this_number_scales]=np.nanvar(incrementsy,1);
    
    # Sur alpha
    alpha_hat=np.nan*np.zeros(shape = (len(x_hat)));
    alpha0=np.angle(np.diff(x_hat)+ 1j*np.diff(y_hat) )
    alpha_hat[0:len(alpha0)]=alpha0
    # alpha_hat=np.unwrap(alpha_hat)
    scales_lista, incrementsa, scales_list_in_points=compute_increments_for_all_scales(np.unwrap(alpha_hat),fps);
    scales_list_global_alpha[kk,:this_number_scales]=scales_lista;
    increments_global_alpha[kk,:this_number_scales,:this_inc_length]=incrementsa;
    Q2_list_global_alpha[kk,:this_number_scales]=np.nanvar(incrementsa,1);
    
    # del scales_list, increments, scales_list_in_points
    
    if kk==0:
        alpha_total_=alpha_hat
        y_hat_total_=y_hat
        x_hat_total_=x_tot_table[kk]
        x_total_=x_hat
    else:
            alpha_total_=np.concatenate((alpha_total_, alpha_hat), axis=0)
            y_hat_total_=np.concatenate((y_hat_total_, y_hat), axis=0)
            x_hat_total_=np.concatenate((x_hat_total_, x_hat), axis=0)
            x_total_=np.concatenate((x_total_, x_tot_table[kk]), axis=0)
    if kk==longest_index:
        plt.figure(100)
        plt.subplot(211)
        plt.plot(x_hat,y_hat, color=color_for_data,alpha=0.75)
        plt.subplot(223)
        plt.plot(t_tot_table[longest_index],x_hat, color=color_for_data,alpha=0.75)
        plt.subplot(224)
        plt.plot(t_tot_table[longest_index],y_hat, color=color_for_data,alpha=0.75)

# %
            
Hist_alpha_x,Hist_alpha_y=histogram_norm_marc(alpha_total_,nbins)
Hist_y_hat_x,Hist_y_hat_y=histogram_norm_marc(y_hat_total_,nbins)
Hist_x_hat_x,Hist_x_hat_y=histogram_norm_marc(x_hat_total_,nbins)
Hist_x_x,Hist_x_y=histogram_norm_marc(x_total_,nbins)

scales_list_global=scales_list_global_x
scale_list_average=np.nanmean(scales_list_global_x,0) 

Q_2_x0_global=np.nanvar(increments_global_x0,2)
Q_2_x_global=np.nanvar(increments_global_x,2)
Q_2_y_global=np.nanvar(increments_global_y,2)
Q_2_alpha_global=np.nanvar(increments_global_alpha,2)

Q_2_average_x0=np.nanmean(Q_2_x0_global ,0 )
Q_2_average_x=np.nanmean(Q_2_x_global ,0 )
Q_2_average_y=np.nanmean(Q_2_y_global ,0 )
Q_2_average_alpha=np.nanmean(Q_2_alpha_global ,0 )

Q_2_average_x0_err=np.nanstd(Q_2_x0_global ,0 )
Q_2_average_x_err=np.nanstd(Q_2_x_global ,0 )
Q_2_average_y_err=np.nanstd(Q_2_y_global ,0 )
Q_2_average_alpha_err=np.nanstd(Q_2_alpha_global ,0 )

# %

plt.figure(300)
plt.subplot(221)
plt.plot(Hist_x_x/1e-6,Hist_x_y,linestyle='-',linewidth=3, color=color_for_data,alpha=0.75)
plt.xlabel(r'$x$ $(\mu $m$)$')
plt.ylabel(r'P$(x)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(222)
plt.plot(Hist_y_hat_x/1e-6,Hist_y_hat_y,linestyle='-',linewidth=3, color=color_for_data,alpha=1)
plt.xlabel(r'$y$ $(\mu $m$)$')
plt.ylabel(r'P$(y)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(223)
plt.plot(Hist_x_hat_x/1e-6,Hist_x_hat_y,linestyle='-',linewidth=3, color=color_for_data,alpha=1)
plt.xlabel(r'$\hat x$ $(\mu $m$)$')
plt.ylabel(r'P$(\hat x)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(224)
plt.plot(Hist_alpha_x/2/np.pi,Hist_alpha_y,linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
plt.xlabel(r'$\alpha$ $(2 \pi)$')
plt.ylabel(r'P$(\alpha)$')
plt.yscale('log')
# plt.xlim([-0.5,0.5])
plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()


plt.figure(40000)


plt.subplot(222)
plt.plot(Hist_alpha_x/2/np.pi,Hist_alpha_y,linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
plt.xlabel(r'$\alpha$ $(2 \pi)$')
plt.ylabel(r'P$(\alpha)$')
# plt.yscale('log')
# plt.xlim([-0.5,0.5])
# plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()
# %
# with open('figure_1_1.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['t','x', 'y','t_hat','x_hat'])  # header
#     writer.writerows(zip(t_data, x_data, y_data,t_hat_data,x_hat_data))

# % MSD sur x_hat et y_hat
plt.figure(400)

for kk in range(0,number_traj,1):
    plt.subplot(221)
    plt.plot(scales_list_global[kk],Q_2_x_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(223)
    plt.plot(scales_list_global[kk],Q_2_y_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(122)
    plt.plot(scales_list_global[kk],(Q_2_x_global[kk])/Q_2_y_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)

plt.subplot(221)
plt.plot(scale_list_average,Q_2_average_x,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(223)
plt.plot(scale_list_average,Q_2_average_y,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[y]}$ $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.plot(scale_list_average, 1/12*(width-5e-6)*(width+5e-6)+0*scale_list_average,linestyle='--',linewidth=2,label=legend_for_data, color=[0,0,0],alpha=0.25)
plt.plot(scale_list_average, 1/12*(width-3e-6)*(width+5e-6)+0*scale_list_average,linestyle='--',linewidth=2,label=legend_for_data, color=[0,0,0],alpha=0.25)

plt.subplot(122)

plt.plot(scale_list_average,Q_2_average_x/Q_2_average_y,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}/Q_2^{[y]}$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.tight_layout()
# %
plt.figure(500)
for kk in range(0,number_traj,1):
    plt.subplot(222)
    plt.plot(scales_list_global[kk],Q_2_alpha_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(224)
    plt.plot(scales_list_global[kk],Q_2_alpha_global[kk]/scales_list_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)

plt.subplot(121)
plt.plot(Hist_alpha_x/2/np.pi,Hist_alpha_y,linestyle='-',label=legend_for_data,linewidth=3, color=color_for_data,alpha=1)
plt.xlabel(r'$\alpha$ $(2 \pi)$')
plt.ylabel(r'P$(\alpha)$')
plt.yscale('log')
plt.xlim([-0.5,0.5])
plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(222)
plt.plot(scale_list_average,Q_2_average_alpha,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\alpha]}$ ')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)


plt.subplot(224)
plt.plot(scale_list_average,Q_2_average_alpha/scale_list_average,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)

plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\alpha]}/\tau$ ')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)



plt.figure(40001)

plt.plot(scale_list_average,1/(Q_2_average_alpha/2/np.pi/2/np.pi/scale_list_average),linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$1/(Q_2^{[\alpha]}/\tau)$ $($s$)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)



# plt.tight_layout()

# %

plt.figure(600)

for kk in range(0,number_traj,1):
    plt.subplot(121)
    plt.plot(scales_list_global[kk],Q_2_x_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(122)
    plt.plot(scales_list_global[kk],Q_2_y_global[kk],linestyle=':',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
plt.subplot(121)
plt.plot(scale_list_average,Q_2_average_x,linestyle='-',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(122)
plt.plot(scale_list_average,Q_2_average_y,linestyle=':',linewidth=5,label=legend_for_data, color=color_for_data,alpha=0.75)


plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[y]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)


Q_2_average_x0=np.nanmean(Q_2_x0_global ,0 )
Q_2_average_x=np.nanmean(Q_2_x_global ,0 )
Q_2_average_y=np.nanmean(Q_2_y_global ,0 )
Q_2_average_alpha=np.nanmean(Q_2_alpha_global ,0 )

Q_2_average_x0_err=np.nanstd(Q_2_x0_global ,0 )
Q_2_average_x_err=np.nanstd(Q_2_x_global ,0 )
Q_2_average_y_err=np.nanstd(Q_2_y_global ,0 )
Q_2_average_alpha_err=np.nanstd(Q_2_alpha_global ,0 )


# with open(os.path.join(target_folder,'figure_2.csv'), 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['tau','MSD x','MSD x err', 'MSD y','MSD y err','MSD x_hat','MSD x_hat err','MSD phi_hat','MSD phi_hat err'])  # header
#     writer.writerows(zip(scale_list_average, Q_2_average_x0, Q_2_average_x0_err, Q_2_average_y, Q_2_average_y_err, Q_2_average_x, Q_2_average_x_err, Q_2_average_alpha, Q_2_average_alpha_err))


# %
# %
plt.figure(700)

for kk in range(0,number_traj,1):
    plt.subplot(221)
    plt.plot(scales_list_global[kk],Q_2_x_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(222)
    plt.plot(scales_list_global[kk],Q_2_y_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(223)
    plt.plot(scales_list_global[kk],np.sqrt(Q_2_x_global[kk]/scales_list_global[kk]/scales_list_global[kk]),linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)
    plt.subplot(224)
    plt.plot(scales_list_global[kk],Q_2_x_global[kk]/scales_list_global[kk],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=0.25)

plt.subplot(221)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(222)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[y]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(223)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$\sqrt{Q_2^{[\hat x]}/\tau^2}$  $($m$/$s$)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(224)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}/\tau$  $($m$^2/$s$)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

# %
TA_list=[]
TA_err_list=[]
V0_list_=[]
V0y_list_=[]
Dx_list_=[]
V0_list_err=[]
V0y_list_err=[]

D_list_err=[]
D_alpha_list_=[]
D_alpha_list_err=[]
D_alpha_fit_range=(scale_list_average>0.25)*(scale_list_average<2)
V0_fit_range=(scale_list_average>0.0)*(scale_list_average<0.25)
D_fit_range=(scale_list_average>0.75)*(scale_list_average<5)
V0_fit_range=(scale_list_average>0.03)*(scale_list_average<0.3)
D_fit_range=(scale_list_average>0.75)*(scale_list_average<5)
D_fit_range=(scale_list_average>5)*(scale_list_average<20) # dans le cas des mesures directes

# %
for kk in range(0,number_traj,1):
    plt.figure(700)
    plt.subplot(221)
    plt.plot(scales_list_global[kk][V0_fit_range],Q_2_x_global[kk][V0_fit_range],linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
    plt.subplot(222)
    plt.plot(scales_list_global[kk][V0_fit_range],Q_2_y_global[kk][V0_fit_range],linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
    plt.subplot(223)
    plt.plot(scales_list_global[kk][V0_fit_range],np.sqrt(Q_2_x_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]),linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
    plt.subplot(221)
    plt.plot(scales_list_global[kk][D_fit_range],Q_2_x_global[kk][D_fit_range],linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
    plt.subplot(224)
    plt.plot(scales_list_global[kk][D_fit_range],Q_2_x_global[kk][D_fit_range]/scales_list_global[kk][D_fit_range],linestyle='-',linewidth=3,label=legend_for_data, color=color_for_data,alpha=1)
    plt.figure(500)
    plt.subplot(222)
    plt.plot(scales_list_global[kk][D_alpha_fit_range],Q_2_alpha_global[kk][D_alpha_fit_range],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=1)
    plt.subplot(224)
    plt.plot(scales_list_global[kk][D_alpha_fit_range],Q_2_alpha_global[kk][D_alpha_fit_range]/scales_list_global[kk][D_alpha_fit_range],linestyle='-',linewidth=2,label=legend_for_data, color=color_for_data,alpha=1)


    V0_list_=V0_list_+ [np.nanmean(np.sqrt(Q_2_x_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]))]
    Dx_list_=Dx_list_+ [np.nanmean(0.5*Q_2_x_global[kk][D_fit_range]/scales_list_global[kk][D_fit_range])]
    V0_list_err=V0_list_err+ [np.nanstd(np.sqrt(Q_2_x_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]))]
    D_list_err=D_list_err+ [np.nanstd(0.5*Q_2_x_global[kk][D_fit_range]/scales_list_global[kk][D_fit_range])]
    D_alpha_list_err=D_alpha_list_err+ [np.nanstd(Q_2_alpha_global[kk][D_alpha_fit_range]/scales_list_global[kk][D_alpha_fit_range])]
    D_alpha_list_=D_alpha_list_+ [np.nanmean(Q_2_alpha_global[kk][D_alpha_fit_range]/scales_list_global[kk][D_alpha_fit_range])]
    TA_list=TA_list+[np.nanmean(Q_2_x_global[kk][V0_fit_range]/Q_2_y_global[kk][V0_fit_range])]
    TA_err_list=TA_err_list+[np.nanstd(Q_2_x_global[kk][V0_fit_range]/Q_2_y_global[kk][V0_fit_range])]
    
    V0y_list_=V0y_list_+ [np.nanmean(np.sqrt(Q_2_y_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]))]
    V0y_list_err=V0y_list_err+ [np.nanstd(np.sqrt(Q_2_y_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]/scales_list_global[kk][V0_fit_range]))]
V0_set_average=np.nanmean(V0_list_)
V0_set_average_err=np.std(V0_list_)+np.nanmean(V0_list_err)
V0y_set_average=np.nanmean(V0y_list_)
V0y_set_average_err=np.std(V0y_list_)+np.nanmean(V0y_list_err)

D_set_average=np.nanmean(Dx_list_)
D_set_average_err=np.std(Dx_list_)+np.nanmean(D_list_err)
D_alpha_set_average=np.nanmean(D_alpha_list_)
D_alpha_set_average_err=np.std(D_alpha_list_)+np.nanmean(D_alpha_list_err)

V0x_set_average_list=V0x_set_average_list + [V0_set_average]
V0x_set_average_err_list=V0x_set_average_err_list + [V0_set_average_err]
V0y_set_average_list=V0y_set_average_list + [V0y_set_average]
V0y_set_average_err_list=V0y_set_average_err_list + [V0y_set_average_err]

D_set_average_list=D_set_average_list + [D_set_average]
D_set_average_err_list=D_set_average_err_list + [D_set_average_err]
D_alpha_set_average_list=D_alpha_set_average_list + [D_alpha_set_average]
D_alpha_set_average_err_list=D_alpha_set_average_err_list + [D_alpha_set_average_err]

TA_set_average=np.nanmean(TA_list)
TA_set_average_err=np.nanstd(TA_list)+np.nanmean(TA_err_list)


TA_estimate_list=TA_estimate_list+[TA_set_average]
TA_estimate_list_err=TA_estimate_list_err+[TA_set_average_err]

A_list=A_list+[Amp]
T_list=T_list+[Period]


plt.figure(800)

plt.subplot(121)
plt.plot(Amp,V0_set_average,'*',linewidth=3, color=color_for_data,alpha=0.5)

plt.errorbar(Amp,V0_set_average, yerr=V0_set_average_err, xerr=None,linewidth=3, color=color_for_data,alpha=0.5)
plt.xlabel(r'$A$ $($mbar$)$')
plt.ylabel(r'$V_0$ $($m$/$s$)$')

# plt.yscale('log')
# plt.xlim([-0.5,0.5])
# plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(122)
plt.plot(Amp,D_set_average,'*',linewidth=3, color=color_for_data,alpha=0.5)

plt.errorbar(Amp,D_set_average, yerr=D_set_average_err, xerr=None,linewidth=3, color=color_for_data,alpha=0.5)
plt.xlabel(r'$A$ $($mbar$)$')
plt.ylabel(r'$D_x$ $($m$^2/$s$)$')
# plt.yscale('log')
# plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)


plt.figure(800)

plt.subplot(121)
plt.plot(Amp,V0_set_average,'*',linewidth=3, color=color_for_data,alpha=0.5)

plt.errorbar(Amp,V0_set_average, yerr=V0_set_average_err, xerr=None,linewidth=3, color=color_for_data,alpha=0.5)
plt.xlabel(r'$A$ $($mbar$)$')
plt.ylabel(r'$V_0$ $($m$/$s$)$')

# plt.yscale('log')
# plt.xlim([-0.5,0.5])
# plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(122)
plt.plot(Amp,D_set_average,'*',linewidth=3, color=color_for_data,alpha=0.5)

plt.errorbar(Amp,D_set_average, yerr=D_set_average_err, xerr=None,linewidth=3, color=color_for_data,alpha=0.5)
plt.xlabel(r'$A$ $($mbar$)$')
plt.ylabel(r'$D_x$ $($m$^2/$s$)$')
# plt.yscale('log')
# plt.legend()
plt.grid(color='k', linestyle='-', linewidth=0.5)
# %
V0=100e-6
Tau=1


def Brownian_MSD_prediction(tau_list,V0,tau_v):
    nu=1/tau_v
    Q_2_test=2*( V0**2/nu)* (tau_list-tau_v*(1-np.exp(-nu*tau_list)))
    return(Q_2_test)

def Brownian_MSD_confined_prediction(tau_list,V0,tau_v,A):
    nu=1/tau_v
    Q_2_test=0*tau_list
    Q_2_test[(1 - np.exp(-nu*tau_list ))<0.90]=( 2*V0**2/nu)*tau_list[(1 - np.exp(-nu*tau_list ))<0.90]**2
    Q_2_test[(1 - np.exp(-nu*tau_list ))>0.80]=A*(1-np.exp(-nu*tau_list[(1 - np.exp(-nu*tau_list ))>0.80]))
    return(Q_2_test)


kk=longest_index

V0x_fit_list=[]
V0y_fit_list=[]
Taux_fit_list=[]
Tauy_fit_list=[]
Dx_fit_list=[]
Dy_fit_list=[]
TA_fit_list=[]

plt.figure(900)

for kk in range(0,number_traj,1):
    try:
        V0_x=np.nan
        V0_y=np.nan
        tau_v_x=np.nan
        tau_v_y=np.nan

        scales_list_=scales_list_global[kk]
        Q_2_x_=Q_2_x_global[kk]
        Q_2_y_=Q_2_y_global[kk]
        Q_2_x0_=Q2_list_global_x0[kk]
        scales_list_l=scales_list_global[longest_index][-1]
        
        fit_range_y0=(scales_list_>1)#*(scale_list_average<10)
        fit_range_y0[np.isnan(scales_list_)]=True
        fit_range_y0[0]=True
        
        fit_range_y= np.invert(fit_range_y0)
        p_opt2,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_y],Q_2_y_[fit_range_y], p0=(V0,Tau))
        Q2_fit2=Brownian_MSD_prediction(scales_list_,*p_opt2)
        [V0_y,tau_v_y]=p_opt2
        V0y_fit_list=V0y_fit_list+ [V0_y]
        Tauy_fit_list=Tauy_fit_list+ [tau_v_y]
        
        plt.subplot(122)
        plt.plot(scales_list_,Q_2_y_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
        plt.plot(scales_list_[fit_range_y],Q_2_y_[fit_range_y], color=color_for_data,linewidth=2,alpha=0.75)
        plt.plot(scales_list_,Q2_fit2,'--', color=color_for_data,linewidth=1,alpha=0.75)
        
        try:
            fit_range_x=scales_list_<scales_list_l/20
            fit_range_x[0:2]=False
            fit_range_x[np.isnan(scales_list_)]=False
        
            p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
            Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
            [V0_x,tau_v_x]=p_opt1
        except:
            print('No luck')
            try:
                fit_range_x=scales_list_<scales_list_l/40
                fit_range_x[0:2]=False
                fit_range_x[np.isnan(scales_list_)]=False
        
                p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
                Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
                [V0_x,tau_v_x]=p_opt1
            except:
                print('No luck')
                try:
                    fit_range_x=scales_list_<scales_list_l/60
                    fit_range_x[0:2]=False
                    fit_range_x[np.isnan(scales_list_)]=False
        
                    p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
                    Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
                    [V0_x,tau_v_x]=p_opt1
                except:
                    print('No luck')
                    try:
                        fit_range_x=scales_list_<scales_list_l/80
                        fit_range_x[0:2]=False
                        fit_range_x[np.isnan(scales_list_)]=False
            
                        p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
                        Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
                        [V0_x,tau_v_x]=p_opt1
                    except:
                        print('No luck')
    except:
        print('sad')
    V0x_fit_list=V0x_fit_list+ [V0_x]
    Taux_fit_list=Taux_fit_list+ [tau_v_x]

    Dx_fit_list=Dx_fit_list+[0.5*V0_x*V0_x/tau_v_x]
    Dy_fit_list=Dy_fit_list+[0.5*V0_y*V0_y/tau_v_y]
    TA_fit_list=TA_fit_list+[(0.5*V0_x*V0_x/tau_v_x)/(0.5*V0_y*V0_y/tau_v_y)-1]
    try:
        plt.subplot(121)
        plt.plot(scales_list_,Q_2_x_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
        plt.plot(scales_list_[fit_range_x],Q_2_x_[fit_range_x], color=color_for_data,linewidth=2,alpha=0.75)
        plt.plot(scales_list_,Q2_fit,'--', color=color_for_data,linewidth=1,alpha=0.75)
    except:
        print('sad')

plt.subplot(122)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2$ $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(121)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2$ $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()

# %

TA_fit_set_average_fit=np.nanmean(TA_fit_list)
TA_fit_set_average_fit_err=np.nanstd(TA_fit_list)

V0x_fit_set_average_fit=np.nanmean(V0x_fit_list)
V0x_fit_set_average_fit_err=np.nanstd(V0x_fit_list)
V0y_fit_set_average_fit=np.nanmean(V0y_fit_list)
V0y_fit_set_average_fit_err=np.nanstd(V0y_fit_list)

Dx_fit_set_average=np.nanmean(Dx_fit_list)
Dx_fit_set_average_err=np.nanstd(Dx_fit_list)
Dy_fit_set_average=np.nanmean(Dy_fit_list)
Dy_fit_set_average_err=np.nanstd(Dy_fit_list)

Taux_fit_set_average=np.nanmean(Taux_fit_list)
Taux_fit_set_average_err=np.nanstd(Taux_fit_list)
Tauy_fit_set_average=np.nanmean(Tauy_fit_list)
Tauy_fit_set_average_err=np.nanstd(Tauy_fit_list)

# %
V0x_fit_set_average_fit_list=V0x_fit_set_average_fit_list+ [V0x_fit_set_average_fit]
V0y_fit_set_average_fit_list=V0y_fit_set_average_fit_list+[V0y_fit_set_average_fit]
Dx_fit_set_average_fit_list=Dx_fit_set_average_fit_list+[Dx_fit_set_average]
Dy_fit_set_average_fit_list=Dy_fit_set_average_fit_list+[Dy_fit_set_average]
Taux_fit_estimate_list=Taux_fit_estimate_list+[Taux_fit_set_average]
Tauy_fit_estimate_list=Tauy_fit_estimate_list+[Tauy_fit_set_average]
TA_fit_estimate_list=TA_fit_estimate_list+[TA_fit_set_average_fit]

V0x_fit_set_average_fit_list_err=V0x_fit_set_average_fit_list_err+ [V0x_fit_set_average_fit]
V0y_fit_set_average_fit_list_err=V0y_fit_set_average_fit_list_err+[V0y_fit_set_average_fit]
Dx_fit_set_average_fit_list_err=Dx_fit_set_average_fit_list_err+[Dx_fit_set_average]
Dy_fit_set_average_fit_list_err=Dy_fit_set_average_fit_list_err+[Dy_fit_set_average]
Taux_fit_estimate_list_err=Taux_fit_estimate_list_err+[Taux_fit_set_average]
Tauy_fit_estimate_list_err=Tauy_fit_estimate_list_err+[Tauy_fit_set_average]
TA_fit_estimate_list_err=TA_fit_estimate_list_err+[TA_fit_set_average_fit]

# %
x_ech_tot_table={}
y_ech_tot_table={}
t_ech_tot_table={}


length_table=[]
visible=0
true_kij=0
index_mute=0

new_period=1*Period

for kii in range(0, len(x_tot_table), 1):
            t_ech_,x_ech_,y_ech_,new_fs=undersample_Marc(t_tot_table[true_kij],x_tot_table[true_kij],y_tot_table[true_kij],new_period,fs,visible)
            for tt in range(len(t_ech_)):
                
                x_ech_tot_table[index_mute]=x_ech_[tt,:]
                y_ech_tot_table[index_mute]=y_ech_[tt,:]
                t_ech_tot_table[index_mute]=t_ech_[tt,:]
                length_table.append(len(x_ech_tot_table[index_mute]))
                
                index_mute=index_mute+1
            true_kij=true_kij+1

nbr_traj=len(t_tot_table) # total traj number
print(nbr_traj)
longest_index=np.argmax(length_table)
# get the lengths of each trajectory :
# %
data0=x_ech_tot_table[longest_index]; # in case there are two
scales_list_ech0, increments_all_scales_ech0 , scales_list_in_points_ech0=compute_increments_for_all_scales(data0,new_fs)
Q_2_average_ech0=np.nanvar(increments_all_scales_ech0 ,1 )

# %
number_scales_ech=len(scales_list_ech0)
number_traj_ech=len(x_ech_tot_table)
L_inc_ech=len(increments_all_scales_ech0[0])

scales_list_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
increments_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech,L_inc_ech));
scales_list_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
increments_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech,L_inc_ech));

Q2_list_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
Q2_list_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));

# %
for kk in range(number_traj_ech):
    scales_listx_ech, incrementsx_ech, scales_list_in_points_ech=compute_increments_for_all_scales(x_ech_tot_table[kk],new_fs);
    try:
        this_inc_length=len(incrementsx_ech[0])
        this_number_scales=len(scales_listx_ech)
        scales_list_global_x_ech[kk,:this_number_scales]=scales_listx_ech;
        increments_global_x_ech[kk,:this_number_scales,:this_inc_length]=incrementsx_ech;
        Q2_list_global_x_ech[kk,:this_number_scales]=np.nanvar(incrementsx_ech,1);
        y_hat=y_ech_tot_table[kk]
        x_hat=x_ech_tot_table[kk]
        scales_listy_ech, incrementsy_ech, scales_list_in_points_ech=compute_increments_for_all_scales(y_hat,new_fs);
        scales_list_global_y_ech[kk,:this_number_scales]=scales_listy_ech;
        increments_global_y_ech[kk,:this_number_scales,:this_inc_length]=incrementsy_ech;
        Q2_list_global_y_ech[kk,:this_number_scales]=np.nanvar(incrementsy_ech,1);
        
        if kk==0:
            y_hat_total_=y_hat
            x_hat_total_=x_ech_tot_table[kk]
            x_total_=x_hat
        else:
                y_hat_total_=np.concatenate((y_hat_total_, y_hat), axis=0)
                x_hat_total_=np.concatenate((x_hat_total_, x_hat), axis=0)
                x_total_=np.concatenate((x_total_, x_ech_tot_table[kk]), axis=0)
    except:
            print("Nope") 
# %
            
Hist_y_hat_x,Hist_y_hat_y=histogram_norm_marc(y_hat_total_,nbins)
Hist_x_hat_x,Hist_x_hat_y=histogram_norm_marc(x_hat_total_,nbins)
Hist_x_x,Hist_x_y=histogram_norm_marc(x_total_,nbins)

scales_list_global_ech=scales_list_global_x_ech
scale_list_average_ech=np.nanmean(scales_list_global_x_ech,0) 

Q_2_x_global_ech=np.nanvar(increments_global_x_ech,2)
Q_2_y_global_ech=np.nanvar(increments_global_y_ech,2)

Q_2_average_x_ech=np.nanmean(Q_2_x_global_ech ,0 )
Q_2_average_y_ech=np.nanmean(Q_2_y_global_ech ,0 )
# %

if Amp==0:
    color_for_data2=0*np.array(color_for_data)
else:
    color_for_data2=0*np.array(color_for_data)
    color_for_data2[0]=1.0-0.5*color_for_data[0]
    color_for_data2[1]=1.0-0.5*color_for_data[1]
    color_for_data2[2]=1.0-0.5*color_for_data[2]
    color_for_data2[3]=1

plt.figure(300)
plt.subplot(221)
plt.plot(Hist_x_x/1e-6,Hist_x_y,'*',linestyle='none',linewidth=3, color=color_for_data2,alpha=0.75)
plt.xlabel(r'$x$ $(\mu $m$)$')
plt.ylabel(r'P$(x)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(222)
plt.plot(Hist_y_hat_x/1e-6,Hist_y_hat_y,'*',linestyle='none',linewidth=3, color=color_for_data2,alpha=1)
plt.xlabel(r'$y$ $(\mu $m$)$')
plt.ylabel(r'P$(y)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(223)
plt.plot(Hist_x_hat_x/1e-6,Hist_x_hat_y,'*',linestyle='none',linewidth=3, color=color_for_data2,alpha=1)
plt.xlabel(r'$\hat x$ $(\mu $m$)$')
plt.ylabel(r'P$(\hat x)$')
# plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.figure(400)
plt.subplot(221)
plt.plot(scale_list_average_ech,Q_2_average_x_ech,'*',linestyle='none',linewidth=5,label=legend_for_data, color=color_for_data2,alpha=1)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(223)
plt.plot(scale_list_average_ech,Q_2_average_y_ech,'*',linestyle='none',linewidth=5,label=legend_for_data, color=color_for_data2,alpha=1)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[y]}$ $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

plt.subplot(122)
plt.plot(scale_list_average_ech,Q_2_average_x_ech/Q_2_average_y_ech,'*',linestyle='none',linewidth=5,label=legend_for_data, color=color_for_data2,alpha=1)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}/Q_2^{[y]}$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()

plt.figure(600)

for kk in range(0,number_traj_ech,1):
    plt.subplot(121)
    plt.plot(scales_list_global_ech[kk],Q_2_x_global_ech[kk],linestyle='--',linewidth=0.5,label=legend_for_data, color=color_for_data2,alpha=1)


plt.subplot(121)
plt.plot(scale_list_average_ech,Q_2_average_x_ech,'*',linestyle='none',linewidth=5,label=legend_for_data, color=color_for_data2,alpha=1)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[\hat x]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.subplot(122)

plt.plot(scale_list_average_ech,Q_2_average_y_ech,'*',linestyle='none',linewidth=5,label=legend_for_data, color=color_for_data2,alpha=1)
plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$Q_2^{[y]}$  $($m$^2)$')
plt.xscale('log')
plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)

D_ech_list_=[]
D_list_err=[]
D_fit_range_ech=(scales_list_global_ech[0]>1/new_fs*0.75)*(scales_list_global_ech[0]<10)

for kk in range(0,number_traj_ech,1):

    D_ech_list_=D_ech_list_+ [0.5*np.nanmean(Q_2_x_global_ech[kk][D_fit_range_ech]/scales_list_global_ech[kk][D_fit_range_ech])]
    D_list_err=D_list_err+ [np.nanstd(0.5*Q_2_x_global_ech[kk][D_fit_range_ech]/scales_list_global_ech[kk][D_fit_range_ech])]

D_ech_set_average=np.nanmean(D_ech_list_)
D_ech_set_average_err=np.std(D_ech_list_)+np.nanmean(D_list_err)

D_ech_set_average_list=D_ech_set_average_list + [D_ech_set_average]
D_ech_set_average_err_list=D_ech_set_average_err_list + [D_ech_set_average_err]

longest_index0_backup=longest_index0


tau_alpha_list=2*np.pi*np.pi/np.array(D_alpha_list_)
plt.figure(40001)
plt.plot(tau_alpha_list,'k',label=r'$2\pi ^2 /D_\alpha$')
# plt.plot(Taux_fit_list,'b',label=r'$\tau_x$ fit')
# plt.plot(Tauy_fit_list,'r',label=r'$\tau_y$ fit')
plt.plot(np.array(Dx_list_)/np.array(V0_list_)/np.array(V0_list_),'m',label=r'$V_x^2/D_x$')
plt.plot(tau_alpha_list-np.array(Dx_list_)/np.array(V0_list_)/np.array(V0_list_),'g',label=r'$ 2\pi ^2 /D_\alpha - D_x/V_x^2$')
plt.legend()
# plt.xlabel(r'$\tau$ $($s$)$')
plt.ylabel(r'$\tau$ $($s$)$')
# plt.xscale('log')
# plt.yscale('log')
plt.grid(color='k', linestyle='-', linewidth=0.5)
plt.tight_layout()

# %
save_path=target_folder
# % Saving the results as dataframes
# longest_index0=4
t=t_tot_table[longest_index0]-t_tot_table[longest_index0][0]
x0=x_tot_table[longest_index0]
y=y_tot_table[longest_index0]

# scales_list_=scale_list_average
# Q_2_y_=Q_2_average_y
# Q_2_x_=Q_2_average_x

# fit_range_y= np.invert(fit_range_y0)
# p_opt2,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_y],Q_2_y_[fit_range_y], p0=(V0,Tau))
# Q2_fit2=Brownian_MSD_prediction(scales_list_,*p_opt2)

# fit_range_x=scales_list_<scales_list_l/40
# fit_range_x[0:1]=False
# fit_range_x[np.isnan(scales_list_)]=False

# p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
# Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
# [V0_x,tau_v_x]=p_opt1

# plt.figure(1786469)
# plt.subplot(122)
# plt.plot(scales_list_,Q_2_y_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# plt.plot(scales_list_[fit_range_y],Q_2_y_[fit_range_y], color=color_for_data,linewidth=2,alpha=0.75)
# plt.plot(scales_list_,Q2_fit2,'--', color=color_for_data,linewidth=1,alpha=0.75)


# plt.subplot(124)
# plt.plot(scales_list_,Q_2_x_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# plt.plot(scales_list_[fit_range_x],Q_2_x_[fit_range_x], color=color_for_data,linewidth=2,alpha=0.75)
# plt.plot(scales_list_,Q2_fit,'--', color=color_for_data,linewidth=1,alpha=0.75)

# plt.subplot(122)
# plt.xlabel(r'$\tau$ $($s$)$')
# plt.ylabel(r'$Q_2$ $($m$^2)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.subplot(124)
# plt.xlabel(r'$\tau$ $($s$)$')
# plt.ylabel(r'$Q_2$ $($m$^2)$')
# plt.xscale('log')
# plt.yscale('log')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# plt.tight_layout()
# %
# plt.subplot(221)
# plt.plot(scales_list_,Q_2_y_/scales_list_,'b-',alpha=0.5)

# plt.plot(scales_list_,Q_2_y_/scales_list_/(sum(Q_2_y_/scales_list_)),'g-',alpha=0.5)
# plt.plot(scales_list_,Q_2_y_/scales_list_/scales_list_/(sum(Q_2_y_/scales_list_/scales_list_)),'m-',alpha=0.5)

# plt.plot(scales_list_,Q_2_y_/scales_list_/(sum(Q_2_y_/scales_list_))-Q_2_y_/scales_list_/scales_list_/(sum(Q_2_y_/scales_list_/scales_list_)),'k-',alpha=0.5)
# plt.plot(scales_list_,Q_2_y_/scales_list_/scales_list_/(sum(Q_2_y_/scales_list_/scales_list_))-Q_2_y_/scales_list_/(sum(Q_2_y_/scales_list_)),'k-',alpha=0.5)

# plt.xlabel(r'$ \tau $ $($s$)$')
# plt.ylabel(r'$Q_2 [ y ]$ $(1)$')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid(color='k', linestyle='-', linewidth=0.5)

# plt.subplot(223)
# plt.plot(scales_list_,Q_2_x_/scales_list_,'b-',alpha=0.5)

# plt.plot(scales_list_,Q_2_x_/scales_list_/(sum(Q_2_x_/scales_list_)),'g-',alpha=0.5)
# plt.plot(scales_list_,Q_2_x_/scales_list_/scales_list_/(sum(Q_2_x_/scales_list_/scales_list_)),'m-',alpha=0.5)

# plt.plot(scales_list_,Q_2_x_/scales_list_/(sum(Q_2_x_/scales_list_))-Q_2_x_/scales_list_/scales_list_/(sum(Q_2_x_/scales_list_/scales_list_)),'k-',alpha=0.5)
# plt.plot(scales_list_,Q_2_x_/scales_list_/scales_list_/(sum(Q_2_x_/scales_list_/scales_list_))-Q_2_x_/scales_list_/(sum(Q_2_x_/scales_list_)),'k-',alpha=0.5)

# plt.xlabel(r'$ \tau $ $($s$)$')
# plt.ylabel(r'$Q_2 [ x ]$ $(1)$')
# plt.yscale('log')
# plt.xscale('log')
# plt.grid(color='k', linestyle='-', linewidth=0.5)



# %


# def hist_y_prediction(y_hist_fit_x,A,B,lambda_):
#     y_m=np.median(y_hist_fit_x)
#     size=np.abs(np.max(y_hist_fit_x)-np.min(y_hist_fit_x))
#     y_hist_fit_y=A+0.5*B*np.cosh( lambda_*(y_hist_fit_x-y_m))#/np.cosh( lambda_*size/2)
#     return(y_hist_fit_y)


# y_hist_fit_x=Hist_y_hat_x;
# lambda_=90000;
# y_mean=np.nanmean(y_hist_fit_x);
# A=np.median(Hist_y_hat_y);B=np.median(Hist_y_hat_y);
# y_m=np.median(y_hist_fit_x)
# size=np.abs(np.max(y_hist_fit_x)-np.min(y_hist_fit_x))

# try:
#     p_opt3,p_cov=cf(hist_y_prediction,Hist_y_hat_x[1:-1],Hist_y_hat_y[1:-1], p0=(A,B,lambda_))
#     y_hist_fit_y=hist_y_prediction(y_hist_fit_x,*p_opt3)
# except:
#     print('sad')


# plt.figure()
# plt.plot(Hist_y_hat_x,Hist_y_hat_y,linestyle='-',linewidth=3, color=color_for_data,alpha=1)
# plt.plot(y_hist_fit_x,y_hist_fit_y,linestyle='--',linewidth=3, color=color_for_data,alpha=1)

# # plt.plot([y_hist_fit_x[0],y_hist_fit_x[0]+size],[0,0],linestyle='-',linewidth=3, color=color_for_data,alpha=1)
# # plt.plot([y_m,y_m],[0,np.mean(Hist_y_hat_y)],linestyle='-',linewidth=3, color=color_for_data,alpha=1)

# plt.xlabel(r'$y$ $(\mu $m$)$')
# plt.ylabel(r'P$(y)$')
# # plt.xscale('log')
# # plt.yscale('log')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # %
# from scipy.signal import correlate
# from scipy.signal import correlation_lags


# sig=np.diff(x0[~np.isnan(x0)])*fec
# corr = correlate(sig, sig, mode='full', method='auto')
# lags = 1/fec*correlation_lags(len(sig), len(sig))
# corr /= 2*fec*np.max(corr)

# plt.plot(lags,np.cumsum(corr),'k')



# sig=np.diff(y[~np.isnan(y)])*fec
# corr = correlate(sig, sig, mode='full', method='auto')
# lags = 1/fec*correlation_lags(len(sig), len(sig))
# corr /= 2*fec*np.max(corr)

# plt.plot(lags,np.cumsum(corr)*(np.mean(np.diff(lags))),'r')

# sig=np.diff(x_hat[~np.isnan(x_hat)])*fec
# corr = correlate(sig, sig, mode='full', method='auto')
# lags = 1/fec*correlation_lags(len(sig), len(sig))
# corr /= 2*fec*np.max(corr)

# plt.plot(lags,np.cumsum(corr),'b')
# %





# %
# speedy =y#np.diff(y[~np.isnan(y)])*fec
# corr = correlate(speedy, speedy, mode='full', method='auto')
# lags = 1/fps*correlation_lags(len(speedy), len(speedy))
# corr = corr/corr[lags==0]
# corr_time=0.5/fps*np.cumsum(corr)[-1]
# plt.subplot(122)
# plt.plot(lags,corr,'b')
# plt.plot(lags[np.abs(lags)<corr_time],1+0*lags[np.abs(lags)<corr_time],'k')
# plt.plot([corr_time,corr_time],[0,1],'k')
# plt.plot([-corr_time,-corr_time],[0,1],'k')

# plt.xlabel(r'$\tau$ $($s$)$')
# plt.ylabel(r'C$(\hat x)$')

# print(corr_time)
# print(Taux_fit_set_average)
# print(Tauy_fit_set_average)

# # %%
# print(np.cumsum(corr)[-1])
# print(Taux_fit_set_average)

# # print(Taux_fit_set_average)

# print(Tauy_fit_set_average)

# # print(Tauy_fit_set_average)


# # %%
# plt.plot(Taux_fit_list,'Xb')
# plt.plot(Tauy_fit_list,'Pr')

# plt.plot(np.array(V0_list_)*np.array(V0_list_)/np.array(Dx_list_)/4,'^b')

# #     r'Taux_fit_list': Taux_fit_list,
# #     r'Tauy_fit_list': Tauy_fit_list,

# %
Taux_list=np.array(Dx_list_)/np.array(V0_list_)/np.array(V0_list_)
Tauy_list=tau_alpha_list-np.array(Dx_list_)/np.array(V0_list_)/np.array(V0_list_)

t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[longest_index0],x_tot_table[longest_index0],fs,how_many_periods,lag_choice,Frequency_guess,visible)


D_y_estimate=np.array(V0y_list_)*np.array(V0y_list_)*(3+0*np.array(tau_alpha_list))
D_y_est_alpha=np.array(V0y_list_)*np.array(V0y_list_)*(np.array(tau_alpha_list))
D_y_est_2=np.array(V0y_list_)*np.array(V0y_list_)*(np.array(Tauy_list))

D_y_estimate=D_y_est_2

V0_list_=np.array(V0_list_)
V0y_list_=np.array(V0y_list_)

# %


# # save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s"
# curves_to_save = {
#     r'Amplitude': Amp,
#     r'Period': Period,
#     r'scales_list_global':scales_list_global,
#     r'Q_2_x_global': Q_2_x_global,
#     r'Q_2_y_global': Q_2_y_global,
#     r'V0_fit_range': V0_fit_range,
#     r'D_fit_range': D_fit_range,
    
#     r'scale_list_average': scale_list_average,
#     r'Q_2_average_x0': Q_2_average_x0,
#     r'Q_2_average_x': Q_2_average_x,
#     r'Q_2_average_y': Q_2_average_y,
    
    
#         }
# print('saving')

# savemat(os.path.join(save_path,'curves_MSD_supp.mat'), curves_to_save)#{'arr':x})#,'x':x_tot_table,'y':y_tot_table,'mass':mass_tot_table})
# print('save succesfull')
# # %


# %
# save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\2s"
curves_to_save = {
    r'Amplitude': Amp,
    r'Period': Period,
    r't': t, r'x0': x0,
    r'x_hat': x_hat, r'y': y,
    r'Hist_x_x': Hist_x_x, r'Hist_x_y': Hist_x_y,
    r'Hist_y_hat_x': Hist_y_hat_x, r'Hist_y_hat_y': Hist_y_hat_y,
    r'Hist_x_hat_x': Hist_x_hat_x, r'Hist_x_hat_y': Hist_x_hat_y,

    r'scales_list_global': scales_list_global,
    r'Q_2_x0_global': Q_2_x0_global,
    r'Q_2_x_global': Q_2_x_global,
    r'Q_2_y_global': Q_2_y_global,
    
    r'scale_list_average': scale_list_average,
    r'Q_2_average_x0': Q_2_average_x0,
    r'Q_2_average_x': Q_2_average_x,
    r'Q_2_average_y': Q_2_average_y,
    
    r'Q_2_average_x0_err': Q_2_average_x0_err,
    r'Q_2_average_x_err': Q_2_average_x_err,
    r'Q_2_average_y_err': Q_2_average_y_err,
    
    r'scale_list_average_ech': scale_list_average_ech,
    r'Q_2_average_x_ech': Q_2_average_x_ech,
    r'Q_2_average_y_ech': Q_2_average_y_ech,

    # r'scales_list': scales_list_,
    # r'Q2_fitx': Q2_fit,
    # r'Q2_fity': Q2_fit2,
    
    # r'fit_range_x': fit_range_x,
    # r'fit_range_y': fit_range_y,
    
    r'V0_fit_range': V0_fit_range,
    r'D_fit_range': D_fit_range,
    r'D_alpha_range': D_alpha_fit_range,
    r'D_range_ech': D_fit_range_ech,

    # r'Taux_fit_set_average': Taux_fit_set_average,
    # r'Taux_fit_set_average_err': Taux_fit_set_average_err,
    # r'Tauy_fit_set_average': Tauy_fit_set_average,
    # r'Tauy_fit_set_average_err': Tauy_fit_set_average_err,

    # r'Taux_fit_list': Taux_fit_list,
    # r'Tauy_fit_list': Tauy_fit_list,

    # Mesures sur les MSD

    r'V0_set_average': V0_set_average,
    r'V0_set_average_err': V0_set_average_err,
    r'V0y_set_average': V0y_set_average,
    r'V0y_set_average_err': V0y_set_average_err,
    r'Dx_set_average': D_set_average,
    r'Dx_set_average_err': D_set_average_err,
    r'D_ech_set_average': np.nanmean(D_ech_list_),
    r'D_ech_set_average_err': np.nanstd(D_ech_list_),
    r'D_y_est_set_average': np.nanmean(V0y_list_*V0y_list_*Tauy_list),
    r'D_y_est_set_average_err': np.nanstd(V0y_list_*V0y_list_*Tauy_list),
    
    r'V0x_list_': V0_list_,
    r'V0y_list_': V0y_list_,
    r'Dx_list_': Dx_list_,
    r'Dx_ech_list_': D_ech_list_,
    r'D_y_est_list': D_y_est_2,

    r'Taux_list': Taux_list,
    r'Tauy_list': Tauy_list,
    r'Taua_list': tau_alpha_list,
    
    r'Taux_set_average': np.nanmean(Taux_list),
    r'Taux_set_average_err': np.nanstd(Taux_list),
    r'Tauy_set_average': np.nanmean(Tauy_list),
    r'Tauy_set_average_err': np.nanstd(Tauy_list),
    r'Taua_set_average': np.nanmean(tau_alpha_list),
    r'Taua_set_average_err': np.nanstd(tau_alpha_list),

        }
print('saving')

savemat(os.path.join(save_path,'curves_ultimate.mat'), curves_to_save)#{'arr':x})#,'x':x_tot_table,'y':y_tot_table,'mass':mass_tot_table})
print('save succesfull')









print(legend_for_data)







# %%

# plt.figure(1453763243676)
# plt.subplot(231)
# plt.plot(Amp+0*Taux_list , Taux_list,'^',linewidth=3, color=color_for_data,alpha=0.5)

# plt.xscale('log')
# plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$\tau_x$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(232)
# # plt.plot(Amp+0*Tauy_list , Tauy_list,'v',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$\tau_y$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(233)
# # plt.plot(Amp+0*tau_alpha_list , tau_alpha_list,'*',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$\tau_\alpha$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)



# # plt.subplot(4,2,5)
# # plt.plot(Amp+0*tau_alpha_list , V0_list_,'^',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.plot(Amp+0*tau_alpha_list , V0y_list_,'v',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$V$ $($m$/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(4,2,7)
# # plt.plot(Amp+0*tau_alpha_list , np.array(V0_list_)/np.array(V0y_list_),'^',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.plot(Amp+0*tau_alpha_list , np.array(V0y_list_)/np.array(V0y_list_),'v',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$V_x/V_y$ $($1$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)


# # plt.subplot(4,2,6)
# # plt.plot(Amp+0*np.array(Dx_list_) , np.array(Dx_list_),'^',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.plot(Amp+0*np.array(Dx_list_) , np.array(V0y_list_)*np.array(V0y_list_)*np.array(Tauy_list),'v',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$D$ $($m$^2/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(4,2,8)
# # plt.plot(Amp+0*np.array(Dx_list_) , np.array(Dx_list_)/(np.array(V0y_list_)*np.array(V0y_list_)*np.array(Tauy_list)),'^',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.plot(Amp+0*np.array(V0y_list_) , np.array(V0y_list_)*np.array(V0y_list_)*np.array(Tauy_list)/(np.array(V0y_list_)*np.array(V0y_list_)*np.array(Tauy_list)),'v',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$D_x/D_y$ $($1$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # %


# # %



# # plt.figure(8)
# # plt.subplot(121)

# # plt.errorbar(Amp*Amp,np.nanmean(Dx_list_), yerr=np.nanstd(Dx_list_),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_), yerr=np.nanstd(D_ech_list_),marker='*', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$D_\alpha$ $($m$^2/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.plot([0.1,100],[0.1,100],'k--')

# # plt.subplot(122)

# # plt.errorbar(Amp*Amp,np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(D_y_est_alpha), yerr=np.nanstd(D_y_est_alpha),marker='X', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(D_y_est_2), yerr=np.nanstd(D_y_est_2),marker='P', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$D_y$ $($m$^2/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.plot([0.1,100],[0.1,100],'k--')

# # plt.figure(9)
# # plt.subplot(121)

# # plt.errorbar(Amp*Amp,np.nanmean(Dx_list_)/np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(Dx_list_)/np.nanmean(D_y_est_alpha), yerr=np.nanstd(D_y_est_alpha),marker='X', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(Dx_list_)/np.nanmean(D_y_est_2), yerr=np.nanstd(D_y_est_2),marker='P', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)

# # plt.errorbar(Amp*Amp,np.nanmean(np.array(Dx_list_)/np.array(D_y_estimate)), yerr=np.nanstd(np.array(Dx_list_)/np.array(D_y_estimate)),marker='v', xerr=None,linestyle='None',linewidth=3, color=[0,0,1],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(np.array(Dx_list_)/np.array(D_y_est_alpha)), yerr=np.nanstd(np.array(Dx_list_)/np.array(D_y_est_alpha)),marker='X', xerr=None,linestyle='None',linewidth=3, color=[0,0,1],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(np.array(Dx_list_)/np.array(D_y_est_2)), yerr=np.nanstd(np.array(Dx_list_)/np.array(D_y_est_2)),marker='P', xerr=None,linestyle='None',linewidth=3, color=[0,0,1],alpha=0.5)

# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$\langle D_x \rangle /\langle D_y \rangle$ $($m$^2/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.plot([0.1,100],[0.1,100],'k--')

# # plt.subplot(122)

# # plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_)/np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_)/np.nanmean(D_y_est_alpha), yerr=np.nanstd(D_y_est_alpha),marker='X', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)
# # plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_)/np.nanmean(D_y_est_2), yerr=np.nanstd(D_y_est_2),marker='P', xerr=None,linestyle='None',linewidth=3, color=[1,0,0],alpha=0.5)

# # plt.xlabel(r'$A$ $($mbar$)$')
# # plt.ylabel(r'$\langle D_x  / D_y \rangle$ $($m$^2/$s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.plot([0.1,100],[0.1,100],'k--')

# # %

# plt.figure(2)
# plt.subplot(321)
# plt.errorbar(Amp*Amp,np.nanmean(Dx_list_), yerr=np.nanstd(Dx_list_),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_), yerr=np.nanstd(D_ech_list_),marker='*', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A$ $($mbar$)$')
# plt.ylabel(r'$D_x$ $($m$^2/$s$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(323)
# plt.errorbar(Amp*Amp,np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A$ $($mbar$)$')
# plt.ylabel(r'$D_y$ $($m$^2/$s$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(325)
# plt.errorbar(Amp*Amp,np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A$ $($mbar$)$')
# plt.ylabel(r'$D_y$ $($m$^2/$s$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(122)
# plt.errorbar(Amp*Amp,np.nanmean(Dx_list_)/np.nanmean(D_y_estimate), yerr=np.nanstd(Dx_list_/D_y_estimate)/np.nanmean(np.abs(Dx_list_/D_y_estimate)),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(D_ech_list_)/np.nanmean(D_y_estimate), yerr=np.nanstd(D_ech_list_)/np.nanmean(D_y_estimate),marker='*', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(D_y_estimate)/np.nanmean(D_y_estimate), yerr=np.nanstd(D_y_estimate)/np.nanmean(D_y_estimate),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel(r'$A$ $($mbar$)$')
# plt.ylabel(r'$D_x/D_y$ $($1$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')


# plt.figure(3)

# plt.subplot(221)
# plt.errorbar(Amp*Amp,np.nanmean(V0_list_), yerr=np.nanstd(V0y_list_),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$V_y$ $($m$/$s$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(223)
# plt.errorbar(Amp*Amp,np.nanmean(V0y_list_), yerr=np.nanstd(V0y_list_),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$V_y$ $($m$/$s$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(122)
# plt.errorbar(Amp*Amp,np.nanmean(V0_list_)/np.nanmean(V0y_list_), yerr=np.nanstd(V0_list_)/np.nanmean(V0_list_),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(V0y_list_)/np.nanmean(V0y_list_), yerr=np.nanstd(V0y_list_)/np.nanmean(V0y_list_),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)

# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$V_x/V_y$ $($1$)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')


# plt.figure(4)

# plt.subplot(221)
# plt.errorbar(Amp*Amp,np.nanmean(Taux_list), yerr=np.nanstd(Taux_list),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$\tau_x$ $($s$^2)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(223)
# plt.errorbar(Amp*Amp,np.nanmean(Tauy_list), yerr=np.nanstd(Taux_list),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(tau_alpha_list), yerr=np.nanstd(Taux_list),marker='*', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$\tau_x$ $($s$^2)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[0.1,100],'k--')

# plt.subplot(122)
# plt.errorbar(Amp*Amp,np.nanmean(tau_alpha_list)/np.nanmean(tau_alpha_list), yerr=np.nanstd(tau_alpha_list)/np.nanmean(tau_alpha_list),marker='*', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(Taux_list)/np.nanmean(tau_alpha_list), yerr=np.nanstd(Taux_list)/np.nanmean(tau_alpha_list),marker='^', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)
# plt.errorbar(Amp*Amp,np.nanmean(Tauy_list)/np.nanmean(tau_alpha_list), yerr=np.nanstd(Tauy_list)/np.nanmean(tau_alpha_list),marker='v', xerr=None,linestyle='None',linewidth=3, color=color_for_data,alpha=0.5)

# # plt.xscale('log')
# # plt.yscale('log')
# plt.xlabel(r'$A^2$ $($mbar$^2)$')
# plt.ylabel(r'$\tau/\tau_\alpha$ $(1)$')
# plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.plot([0.1,100],[1,0.100],'k--')

# # %%

#   # %

# # nbins=50
# # Hist_y_hat_x,Hist_y_hat_y=histogram_norm_marc(y_hat,nbins)

# # Hist_y_hat_x_t,Hist_y_hat_y_t=histogram_norm_marc(y_hat_total_,nbins)



# # # colors_for_pressures_list = plt.cm.summer(np.linspace(0,1,9))
# # # save_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s"
# # # Ratio=1

# # # target_folder=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\2s\Chalmy_in_chanel_C_tamer_filtre_Benjamin_x5_sinei_1mbar_2s_40fps_long\TIF"
# # # color_for_data=colors_for_pressures_list[Ratio*1]
# # # Amp=1
# # # Period=2
# # # legend_for_data=r'$1$ mbar $2$ s'
# # # Frequency_guess=0.5
# # # how_many_periods=3

# # x_ech_tot_table={}
# # y_ech_tot_table={}
# # t_ech_tot_table={}
# # if Amp==1:
# #     longest_index0=5
# # if Amp==4:
# #     longest_index0=4

# # # longest_index0=longest_index0+1
# # length_table=[]
# # visible=0
# # true_kij=0
# # index_mute=0

# # new_period=1*Period

# # for kii in range(0, len(x_tot_table), 1):
# #             t_ech_,x_ech_,y_ech_,new_fs=undersample_Marc(t_tot_table[true_kij],x_tot_table[true_kij],y_tot_table[true_kij],new_period,fs,visible)
# #             for tt in range(len(t_ech_)):
                
# #                 x_ech_tot_table[index_mute]=x_ech_[tt,:]
# #                 y_ech_tot_table[index_mute]=y_ech_[tt,:]
# #                 t_ech_tot_table[index_mute]=t_ech_[tt,:]
# #                 length_table.append(len(x_ech_tot_table[index_mute]))
                
# #                 index_mute=index_mute+1
# #             true_kij=true_kij+1

# # nbr_traj=len(t_tot_table) # total traj number
# # print(nbr_traj)
# # longest_index=np.argmax(length_table)


# # nbr_traj=len(t_tot_table) # total traj number
# # print(nbr_traj)
# # longest_index=np.argmax(length_table)
# # # get the lengths of each trajectory :
# # # %
# # data0=x_ech_tot_table[longest_index]; # in case there are two
# # scales_list_ech0, increments_all_scales_ech0 , scales_list_in_points_ech0=compute_increments_for_all_scales(data0,new_fs)
# # Q_2_average_ech0=np.nanvar(increments_all_scales_ech0 ,1 )

# # # %
# # number_scales_ech=len(scales_list_ech0)
# # number_traj_ech=len(x_ech_tot_table)
# # L_inc_ech=len(increments_all_scales_ech0[0])

# # scales_list_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
# # increments_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech,L_inc_ech));
# # scales_list_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
# # increments_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech,L_inc_ech));

# # Q2_list_global_x_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));
# # Q2_list_global_y_ech=np.nan*np.zeros(shape = (number_traj_ech,number_scales_ech));

# # # %
# # for kk in range(number_traj_ech):
# #     scales_listx_ech, incrementsx_ech, scales_list_in_points_ech=compute_increments_for_all_scales(x_ech_tot_table[kk],new_fs);
# #     try:
# #         this_inc_length=len(incrementsx_ech[0])
# #         this_number_scales=len(scales_listx_ech)
# #         scales_list_global_x_ech[kk,:this_number_scales]=scales_listx_ech;
# #         increments_global_x_ech[kk,:this_number_scales,:this_inc_length]=incrementsx_ech;
# #         Q2_list_global_x_ech[kk,:this_number_scales]=np.nanvar(incrementsx_ech,1);
# #         y_hat=y_ech_tot_table[kk]
# #         x_hat=x_ech_tot_table[kk]
# #         scales_listy_ech, incrementsy_ech, scales_list_in_points_ech=compute_increments_for_all_scales(y_hat,new_fs);
# #         scales_list_global_y_ech[kk,:this_number_scales]=scales_listy_ech;
# #         increments_global_y_ech[kk,:this_number_scales,:this_inc_length]=incrementsy_ech;
# #         Q2_list_global_y_ech[kk,:this_number_scales]=np.nanvar(incrementsy_ech,1);
        
# #         if kk==0:
# #             y_hat_total_=y_hat
# #             x_hat_total_=x_ech_tot_table[kk]
# #             x_total_=x_hat
# #         else:
# #                 y_hat_total_=np.concatenate((y_hat_total_, y_hat), axis=0)
# #                 x_hat_total_=np.concatenate((x_hat_total_, x_hat), axis=0)
# #                 x_total_=np.concatenate((x_total_, x_ech_tot_table[kk]), axis=0)
# #     except:
# #             print("Nope") 
# # # %
            
# # Hist_y_hat_x,Hist_y_hat_y=histogram_norm_marc(y_hat_total_,nbins)
# # Hist_x_hat_x,Hist_x_hat_y=histogram_norm_marc(x_hat_total_,nbins)
# # Hist_x_x,Hist_x_y=histogram_norm_marc(x_total_,nbins)

# # scales_list_global_ech=scales_list_global_x_ech
# # scale_list_average_ech=np.nanmean(scales_list_global_x_ech,0) 

# # Q_2_x_global_ech=np.nanvar(increments_global_x_ech,2)
# # Q_2_y_global_ech=np.nanvar(increments_global_y_ech,2)

# # Q2_x_ech_to_plot=np.nanmean(Q_2_x_global_ech ,0 )
# # Q2_y_ech_to_plot=np.nanmean(Q_2_y_global_ech ,0 )


# # scale_list_ech_to_plot=scale_list_average_ech

# # plt.plot(scale_list_ech_to_plot,Q2_x_ech_to_plot,'*',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)
# # plt.plot(scale_list_ech_to_plot,Q2_y_ech_to_plot,'*',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)

# # # plt.xlabel(r'$\tau$ $($s$)$')
# # # plt.ylabel(r'MSD $($m$^2)$')
# # # plt.xscale('log')
# # # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # # figures pour l'article'

# # kk=longest_index0
# # scales_listx, incrementsx0, scales_list_in_points=compute_increments_for_all_scales(x_tot_table[kk],fps);
# # this_inc_length=len(incrementsx0[0])
# # this_number_scales=len(scales_listx)
# # # scales_list_global_x[kk,:this_number_scales]=scales_listx;
# # increments_global_x0[kk,:this_number_scales,:this_inc_length]=incrementsx0;
# # Q2_list_global_x0[kk,:this_number_scales]=np.nanvar(incrementsx0,1);

# # if kk==longest_index:
# #     visible=1
# # if Amp>0:
# #     t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal(t_tot_table[kk],x_tot_table[kk],fs,how_many_periods,lag_choice,Frequency_guess,visible)
# # # t_mean_list,x_hat,x_mean_std,A_mean_list,freq_mean_list,Phase_mean_list,K_mean_list=demodulate_wavelets_siignal2(t_tot_table[kk],x_tot_table[kk],fs,how_many_periods,lag_choice,Frequency_guess,visible)
# # else:
# #     t_mean_list=t_tot_table[kk]
# #     x_hat=x_tot_table[kk]
    

# # scales_listx, incrementsx, scales_list_in_points=compute_increments_for_all_scales(x_hat,fps);
# # this_inc_length=len(incrementsx[0])
# # this_number_scales=len(scales_listx)
# # scales_list_global_x[kk,:this_number_scales]=scales_listx;
# # increments_global_x[kk,:this_number_scales,:this_inc_length]=incrementsx;
# # Q2_list_global_x[kk,:this_number_scales]=np.nanvar(incrementsx,1);

# # # del scales_listy, incrementsy, scales_list_in_points

# # # Sur y
# # y_hat=y_tot_table[kk]


# # scales_listy, incrementsy, scales_list_in_points=compute_increments_for_all_scales(y_hat,fps);
# # scales_list_global_y[kk,:this_number_scales]=scales_listy;
# # increments_global_y[kk,:this_number_scales,:this_inc_length]=incrementsy;
# # Q2_list_global_y[kk,:this_number_scales]=np.nanvar(incrementsy,1);

# # # # Sur alpha
# # # alpha_hat=np.nan*np.zeros(shape = (len(x_hat)));
# # # alpha0=np.angle(np.diff(x_hat)+ 1j*np.diff(y_hat) )
# # # alpha_hat[0:len(alpha0)]=alpha0
# # # # alpha_hat=np.unwrap(alpha_hat)
# # # scales_lista, incrementsa, scales_list_in_points=compute_increments_for_all_scales(np.unwrap(alpha_hat),fps);
# # # scales_list_global_alpha[kk,:this_number_scales]=scales_lista;
# # # increments_global_alpha[kk,:this_number_scales,:this_inc_length]=incrementsa;
# # # Q2_list_global_alpha[kk,:this_number_scales]=np.nanvar(incrementsa,1);





# # #  fits sur les MSD

# # kk=longest_index0
# # try:
# #     V0_x=np.nan
# #     V0_y=np.nan
# #     tau_v_x=np.nan
# #     tau_v_y=np.nan
# #     scales_list_=scales_list_global[kk]
# #     Q_2_x_=Q_2_x_global[kk]
# #     Q_2_y_=Q_2_y_global[kk]
# #     Q_2_x0_=Q2_list_global_x0[kk]
# #     scales_list_l=scales_list_global[longest_index0][-1]
# #     fit_range_y0=(scales_list_>1)#*(scale_list_average<10)
# #     fit_range_y0[np.isnan(scales_list_)]=True
# #     fit_range_y0[0]=True
# #     fit_range_y= np.invert(fit_range_y0)
# #     p_opt2,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_y],Q_2_y_[fit_range_y], p0=(V0,Tau))
# #     Q2_fit2=Brownian_MSD_prediction(scales_list_,*p_opt2)
# #     [V0_y,tau_v_y]=p_opt2
# #     V0y_fit_list=V0y_fit_list+ [V0_y]
# #     Tauy_fit_list=Tauy_fit_list+ [tau_v_y]

# #     try:
# #         fit_range_x=scales_list_<scales_list_l/10
# #         fit_range_x[scales_list_<1]=False
# #         fit_range_x[scales_list_<1]=True

# #         fit_range_x[np.isnan(scales_list_)]=False
    
# #         p_opt1,p_cov=cf(Brownian_MSD_prediction,scales_list_[fit_range_x],Q_2_x_[fit_range_x], p0=(V0_y,tau_v_y))
# #         Q2_fit=Brownian_MSD_prediction(scales_list_,*p_opt1)
# #         [V0_x,tau_v_x]=p_opt1
# #     except:
# #         print('No luck')

# # except:
# #     print('sad')
# # # %
# # # sous échantillonage
# # t=t_tot_table[longest_index0]-t_tot_table[longest_index0][0]
# # x=x_tot_table[longest_index0]
# # # %
# # # %
# # plt.figure(3721)
# # plt.subplot(211)
# # plt.plot(1e6*x_tot_table[longest_index0],1e6*y_tot_table[longest_index0], color=color_for_data,alpha=0.25)
# # plt.plot(1e6*x_hat,1e6*y_hat, color=color_for_data,alpha=0.75)
# # plt.xlim([0,4500])
# # plt.xlabel(r'$x$ $(\mu$m$)$')
# # plt.ylabel(r'$y$ $(\mu$m$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(223)
# # plt.plot(t,1e6*x_tot_table[longest_index0], color=color_for_data,alpha=0.25)
# # plt.plot(t,1e6*x_hat, color=color_for_data,alpha=0.75)

# # plt.ylabel(r'$x$ $(\mu$m$)$')
# # plt.xlabel(r'$t$ $($s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.subplot(224)
# # plt.plot(t,1e6*y_tot_table[longest_index0], color=color_for_data,alpha=0.25)
# # plt.ylabel(r'$y$ $(\mu$m$)$')
# # plt.xlabel(r'$t$ $($s$)$')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # # %
# # # plt.figure(3722)
# # plt.subplot(299)
# # plt.plot(Hist_y_hat_y_t,1e6*Hist_y_hat_x_t, color=color_for_data,alpha=0.75)
# # # plt.plot(Hist_y_hat_y,Hist_y_hat_x, color=color_for_data,alpha=0.75)
# # plt.ylabel(r'$y$ $(\mu$m$)$')
# # plt.xlabel(r'P$(y)$ $(/\mu$m$)$')
# # # plt.xscale('log')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)


# # # %

# # plt.figure(3722)

# # plt.subplot(121)
# # plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_x0[longest_index0,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.25)
# # plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_x[longest_index0,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.75)
# # plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_y[longest_index0,:],linestyle='--',label=legend_for_data, color=color_for_data,alpha=0.75)

# # plt.plot(scale_list_ech_to_plot,Q2_x_ech_to_plot,'P',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)
# # plt.plot(scale_list_ech_to_plot,Q2_y_ech_to_plot,'X',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)

# # plt.xlabel(r'$\tau$ $($s$)$')
# # plt.ylabel(r'MSD $($m$^2)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.subplot(222)
# # for kk in range(len(scales_list_global_x)):
# #     plt.plot(scales_list_global_x[kk,:],Q2_list_global_x[kk,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.5)


# # plt.xlabel(r'$\tau$ $($s$)$')
# # plt.ylabel(r'MSD$[\hat x]$ $($m$^2)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.tight_layout()

# # plt.subplot(224)
# # for kk in range(len(scales_list_global_x)):
# #     plt.plot(scales_list_global_x[kk,:],Q2_list_global_y[kk,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.5)

# # plt.xlabel(r'$\tau$ $($s$)$')
# # plt.ylabel(r'MSD$[y]$ $($m$^2)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.tight_layout()

# # # %


# # plt.subplot(121)
# # # plt.plot(scales_list_,Q_2_y_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# # plt.plot(scales_list_[fit_range_y],Q_2_y_[fit_range_y], color=color_for_data,linewidth=3,alpha=1)
# # plt.plot(scales_list_,Q2_fit2,'--', color=color_for_data,linewidth=1,alpha=1)


# # plt.xlabel(r'$\tau$ $($s$)$')
# # plt.ylabel(r'MSD$[y]$ $($m$^2)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.tight_layout()





# # plt.subplot(121)
# # # plt.plot(scales_list_,Q_2_x_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# # plt.plot(scales_list_[fit_range_x],Q_2_x_[fit_range_x], color=color_for_data,linewidth=3,alpha=1)
# # plt.plot(scales_list_,Q2_fit,'--', color=color_for_data,linewidth=1,alpha=1)



# # plt.xlabel(r'$\tau$ $($s$)$')
# # plt.ylabel(r'MSD$[y]$ $($m$^2)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.tight_layout()
# # x=x_tot_table[longest_index0]
# # # %%
# # if just_for_show==1:
# #     plt.figure(372120)
    
# #     ax1 = plt.subplot2grid((5,4), (0,0), colspan=3, rowspan=2)
    
# #     plt.plot(1e6*x,1e6*y_hat, color=color_for_data,alpha=0.25)
# #     plt.plot(1e6*x_hat,1e6*y_hat, color=color_for_data,alpha=0.75)
# #     plt.xlim([0,3500])
# #     plt.xlabel(r'$x$ $(\mu$m$)$')
# #     plt.ylabel(r'$y$ $(\mu$m$)$')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
    
# #     ax3 = plt.subplot2grid((5,4), (2,0), colspan=3, rowspan=3)
    
    
# #     plt.plot(t,1e6*x, color=color_for_data,alpha=0.25)
# #     plt.plot(t,1e6*x_hat, color=color_for_data,alpha=0.75)
    
# #     plt.ylabel(r'$x$ $(\mu$m$)$')
# #     plt.xlabel(r'$t$ $($s$)$')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
    
# #     ax4 = plt.subplot2grid((5,4), (2,3), colspan=1, rowspan=3)
    
    
# #     plt.plot(t,1e6*y_hat, color=color_for_data,alpha=0.25)
# #     plt.ylabel(r'$y$ $(\mu$m$)$')
# #     plt.xlabel(r'$t$ $($s$)$')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
    
    
# #     ax2 = plt.subplot2grid((5,4), (0,3), colspan=3 , rowspan=2)
    
# #     plt.plot(1e6*Hist_y_hat_x_t,1e6*Hist_y_hat_y_t, color=color_for_data,alpha=0.75)
# #     # plt.plot(Hist_y_hat_y,Hist_y_hat_x, color=color_for_data,alpha=0.75)
# #     plt.ylabel(r'P$(y)$ $(/\mu$m$)$')
# #     plt.xlabel(r'$y$ $(\mu$m$)$')
# #     # plt.xscale('log')
    
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
# #     # %
    
# #     plt.figure(3721200)
    
# #     ax10 = plt.subplot2grid((4,4), (0,0), colspan=2, rowspan=4)
    
# #     plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_x0[longest_index0,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.25)
# #     plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_x[longest_index0,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.75)
# #     plt.plot(scales_list_global_x[longest_index0,:],Q2_list_global_y[longest_index0,:],linestyle='--',label=legend_for_data, color=color_for_data,alpha=0.75)
    
# #     plt.plot(scale_list_ech_to_plot,Q2_x_ech_to_plot,'P',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)
# #     plt.plot(scale_list_ech_to_plot,Q2_y_ech_to_plot,'X',linestyle='none',label=legend_for_data, color=color_for_data,alpha=1)
    
# #     # plt.plot(scales_list_,Q_2_y_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# #     # plt.plot(scales_list_[fit_range_y],Q_2_y_[fit_range_y], color=color_for_data,linewidth=3,alpha=1)
# #     plt.plot(scales_list_,Q2_fit2,'--', color=color_for_data,linewidth=1,alpha=1)
    
# #     # plt.plot(scales_list_,Q_2_x_, color=color_for_data,linestyle='-',linewidth=1,alpha=0.5)
# #     # plt.plot(scales_list_[fit_range_x],Q_2_x_[fit_range_x], color=color_for_data,linewidth=3,alpha=1)
# #     plt.plot(scales_list_,Q2_fit,'--', color=color_for_data,linewidth=1,alpha=1)
# #     plt.plot(scale_list_average, 1/12*(width-5e-6)*(width+5e-6)+0*scale_list_average,linestyle='--',linewidth=2,label=legend_for_data, color=[0,0,0],alpha=1)
    
# #     plt.xlabel(r'$\tau$ $($s$)$')
# #     plt.ylabel(r'MSD $($m$^2)$')
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
# # else:

# # # %
# #     plt.figure(3721200)

# #     # ax40= plt.subplot2grid((4,4), (0,2), colspan=2, rowspan=2)
# #     plt.subplot(222)
# #     for kk in range(len(scales_list_global_x)):
# #         plt.plot(scales_list_global_x[kk,:],Q2_list_global_x[kk,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.5)
    
# #     plt.xlabel(r'$\tau$ $($s$)$')
# #     plt.ylabel(r'MSD$[\hat x]$ $($m$^2)$')
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
# #     plt.tight_layout()
    
# #     # ax20 = plt.subplot2grid((4,4), (2,2), colspan=2 , rowspan=2)
# #     plt.subplot(224)
# #     for kk in range(len(scales_list_global_x)):
# #         plt.plot(scales_list_global_x[kk,:],Q2_list_global_y[kk,:],linestyle='-',label=legend_for_data, color=color_for_data,alpha=0.5)
    
# #     plt.plot(scale_list_average, 1/12*(width-5e-6)*(width+5e-6)+0*scale_list_average,linestyle='--',linewidth=2,label=legend_for_data, color=[0,0,0],alpha=1)
    
# #     plt.xlabel(r'$\tau$ $($s$)$')
# #     plt.ylabel(r'MSD$[y]$ $($m$^2)$')
# #     plt.xscale('log')
# #     plt.yscale('log')
# #     plt.grid(color='k', linestyle='-', linewidth=0.5)
# #     plt.tight_layout()

# # # %%
# # save_path=target_folder
# # # %% Saving the results as dataframes

# # my_dict = {
# # # Paramètres
# #     r'A_list': A_list,
# #     r'T_list': T_list,
# # # Fits des MSD
# #     r'V0x_fit_set_average_fit_list': V0x_fit_set_average_fit_list, r'V0x_fit_set_average_fit_list_err': V0x_fit_set_average_fit_list_err,
# #     r'V0y_fit_set_average_fit_list': V0y_fit_set_average_fit_list, r'V0y_fit_set_average_fit_list_err': V0y_fit_set_average_fit_list_err,
# #     r'Dx_fit_set_average_fit_list': Dx_fit_set_average_fit_list, r'Dx_fit_set_average_fit_list_err': Dx_fit_set_average_fit_list_err,
# #     r'Dy_fit_set_average_fit_list': Dy_fit_set_average_fit_list, r'Dy_fit_set_average_fit_list_err': Dy_fit_set_average_fit_list_err,
# #     r'Taux_fit_estimate_list': Taux_fit_estimate_list, r'Taux_fit_estimate_list_err': Taux_fit_estimate_list_err,
# #     r'Tauy_fit_estimate_list': Tauy_fit_estimate_list, r'Tauy_fit_estimate_list_err': Tauy_fit_estimate_list_err,
# # # Mesures sur les MSD
# #     r'V0x_set_average_list': V0x_set_average_list, r'V0x_set_average_list_err': V0x_set_average_err_list,
# #     r'V0y_set_average_list': V0y_fit_set_average_fit_list, r'V0y_set_average_list_err': V0y_set_average_err_list,
# #     r'Dx_set_average_list': D_set_average_list, r'Dx_set_average_list_err': D_set_average_err_list,
# #     r'D_alpha_set_average_list': D_alpha_set_average_list, r'D_alpha_set_average_list_err': D_alpha_set_average_err_list,
# # # Mesures sur les MSD
# #     r'D_ech_set_average_list': D_ech_set_average_list, r'D_ech_set_average_err_list': D_ech_set_average_err_list,
# #     }
# # to_save=pd.DataFrame(my_dict)

# # fname_to_save=r"for_articles"+r".csv"
# # print('saving')
# # link_path=os.path.join(save_path, fname_to_save)
# # to_save.to_csv(link_path)
# # print('save succesfull')
# # fname_to_save=r"for_articles"+r".mat"


# # # %%

# # Ratio=3
# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\summup\analysis_results_4s.csv"
# # color_4s=[0,0,0.5]
# # legend_for_4s=r'A $4$ s'
# # puce_A_4s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\summup\analysis_results_2s.csv"
# # color_2s=[0,0.5,0]
# # legend_for_2s=r'A $2$ s'
# # puce_A_2s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\summup\analysis_results_1s.csv"
# # color_1s=[0.5,0,0.5]
# # legend_for_1s=r'A $1$ s'
# # puce_A_1s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\26_02_2025\summup\analysis_results_0p5s.csv"
# # color_0p5s=[0.5,0,0]
# # legend_for_0p5s=r'A $0.5$ s'
# # puce_A_0p5s= pd.read_csv(link_path)


# # Ratio=1
# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\summup\analysis_results_4s.csv"
# # color_4s=[0,0,1]
# # legend_for_4s=r'A $4$ s'
# # puce_A_4s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\summup\analysis_results_2s.csv"
# # color_2s=[0,1,0]
# # legend_for_2s=r'A $2$ s'
# # puce_A_2s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\summup\analysis_results_1s.csv"
# # color_1s=[1,0,1]
# # legend_for_1s=r'A $1$ s'
# # puce_A_1s= pd.read_csv(link_path)

# # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\05_02_2025\summup\analysis_results_0p5s.csv"
# # color_0p5s=[1,0,0]
# # legend_for_0p5s=r'A $0.5$ s'
# # puce_A_0p5s= pd.read_csv(link_path)

# # # %
# # # # # %%


# # # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\summup\analysis_results_4s.csv"
# # # color_4s=[0,0,0.75]
# # # legend_for_4s=r'A $4$ s'
# # # puce_A_4s= pd.read_csv(link_path)

# # # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\summup\analysis_results_2s.csv"
# # # color_2s=[0,0.75,0]
# # # legend_for_2s=r'A $2$ s'
# # # puce_A_2s= pd.read_csv(link_path)

# # # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\summup\analysis_results_1s.csv"
# # # color_1s=[0.75,0,0.75]
# # # legend_for_1s=r'A $1$ s'
# # # puce_A_1s= pd.read_csv(link_path)

# # # link_path=r"\\srvzeus\231_DATA_I1\TheWitcher\PROJET_CLAMI\DATA\fait_par_Marc\oscillatory\19_03_2025\summup\analysis_results_8s.csv"
# # # color_0p5s=[0,0.75,0.75]
# # # legend_for_0p5s=r'A $8$ s'
# # # puce_A_0p5s= pd.read_csv(link_path)
# # # Ratio=1


# # # %



# # # %
# # plt.figure(10)
# # plt.suptitle(r'D mesure renormalisations')
# # plt.subplot(121)
# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'], yerr=puce_A_4s['D_ech_set_average_err_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'], xerr=None,linestyle='None',linewidth=1, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'],'*',linewidth=3, color=color_4s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'], yerr=puce_A_2s['D_ech_set_average_err_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'], xerr=None,linestyle='None',linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'],'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'], yerr=puce_A_1s['D_ech_set_average_err_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'], xerr=None,linestyle='None',linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'],'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'], yerr=puce_A_0p5s['D_ech_set_average_err_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'], xerr=None,linestyle='None',linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'],'*',linewidth=3, color=color_1s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Dx_set_average_list'],'^',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list'],'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Dx_set_average_list'],'^',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Dx_set_average_list'],'^',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['V0y_set_average_list']*puce_A_4s['V0y_set_average_list']*puce_A_4s['Tauy_fit_estimate_list'],'v',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0y_set_average_list']*puce_A_2s['V0y_set_average_list']*puce_A_2s['Tauy_fit_estimate_list'],'v',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['V0y_set_average_list']*puce_A_1s['V0y_set_average_list']*puce_A_1s['Tauy_fit_estimate_list'],'v',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['V0y_set_average_list']*puce_A_0p5s['V0y_set_average_list']*puce_A_0p5s['Tauy_fit_estimate_list'],'v',linewidth=3, color=color_0p5s,alpha=0.5)


# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'ech $D_x$ $($m$^2/$s$)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.subplot(122)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'],'*',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'],'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'],'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'],'*',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'],'^',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'],'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'],'^',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'],'^',linewidth=3, color=color_0p5s,alpha=0.5)


# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'ech $D_x/(V_y^2 \tau_y^0)$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.plot([1,1000],[1,1000],'k--')
# # # %

# # plt.figure(11)
# # plt.subplot(121)


# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_4s['V0y_set_average_list_err']/puce_A_4s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_4s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_4s['V0y_set_average_list_err']/puce_A_4s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_4s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_2s['V0y_set_average_list_err']/puce_A_2s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_2s['V0y_set_average_list_err']/puce_A_2s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_2s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_1s['V0y_set_average_list_err']/puce_A_1s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_1s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_1s['V0y_set_average_list_err']/puce_A_1s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_1s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_0p5s['V0y_set_average_list_err']/puce_A_0p5s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_0p5s['V0y_set_average_list_err']/puce_A_0p5s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_0p5s,alpha=0.5)




# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$D_x/(V_y^2 \tau_y^0)-1$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.plot([1,1000],[1,1000],'k--')


# # plt.subplot(122)


# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(1/puce_A_4s['T_list']*puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(1/puce_A_2s['T_list']*puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(1/puce_A_1s['T_list']*puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(1/puce_A_0p5s['T_list']*puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1,'*',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(1/puce_A_4s['T_list']*puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_4s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(1/puce_A_2s['T_list']*puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(1/puce_A_1s['T_list']*puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(1/puce_A_0p5s['T_list']*puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1,'^',linewidth=3, color=color_0p5s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(1/puce_A_4s['T_list']*puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_4s['V0y_set_average_list_err']/puce_A_4s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_4s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.array(1/puce_A_4s['T_list']*puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_4s['V0y_set_average_list_err']/puce_A_4s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_4s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(1/puce_A_2s['T_list']*puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_2s['V0y_set_average_list_err']/puce_A_2s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.array(1/puce_A_2s['T_list']*puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_2s['V0y_set_average_list_err']/puce_A_2s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_2s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(1/puce_A_1s['T_list']*puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_1s['V0y_set_average_list_err']/puce_A_1s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_1s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.array(1/puce_A_1s['T_list']*puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_1s['V0y_set_average_list_err']/puce_A_1s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_1s,alpha=0.5)

# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(1/puce_A_0p5s['T_list']*puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_0p5s['V0y_set_average_list_err']/puce_A_0p5s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.array(1/puce_A_0p5s['T_list']*puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['Tauy_fit_estimate_list'])-1, yerr=np.array(2*puce_A_0p5s['V0y_set_average_list_err']/puce_A_0p5s['V0y_set_average_list']), xerr=None,linestyle='None',linewidth=1, color=color_0p5s,alpha=0.5)



# # plt.plot([1,1000],[0.2,200],'k--')

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$\frac{1}{T}\cdot D_x/(V_y^2 \tau_y^0)-1$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)


# # # %%

# # plt.figure(12)
# # plt.suptitle(r'Correlation time')
# # plt.subplot(121)

# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Tauy_fit_estimate_list'], yerr=puce_A_1s['Tauy_fit_estimate_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Tauy_fit_estimate_list'],'P',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Taux_fit_estimate_list'], yerr=0.1*puce_A_1s['Taux_fit_estimate_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Tauy_fit_estimate_list'], yerr=puce_A_1s['D_ech_set_average_err_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Tauy_fit_estimate_list'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'o',linewidth=3, color=color_2s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$\tau$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # # plt.xlim([0,1000])
# # # plt.ylim([0.1,10])
# # plt.plot([1,100],[1,0.1],'k--')
# # # %%
# # plt.subplot(122)

# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Tauy_fit_estimate_list'], yerr=puce_A_1s['Tauy_fit_estimate_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Tauy_fit_estimate_list']/puce_A_2s['Taux_fit_estimate_list'],'P',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Taux_fit_estimate_list'], yerr=0.1*puce_A_1s['Taux_fit_estimate_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Taux_fit_estimate_list']/puce_A_2s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Tauy_fit_estimate_list'], yerr=puce_A_1s['D_ech_set_average_err_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Tauy_fit_estimate_list'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['Taux_fit_estimate_list'],'o',linewidth=3, color=color_2s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$\tau_x/\tau_y$ $(1)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')

# # # %

# # # %%

# # plt.figure(13)
# # plt.suptitle(r'swimming spead')
# # plt.subplot(121)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['Dx_set_average_list']/puce_A_2s['Taux_fit_estimate_list']),'X',linewidth=3, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['D_ech_set_average_list']/puce_A_2s['Taux_fit_estimate_list']),'*',linewidth=3, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0x_set_average_list'],'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0x_set_average_list'], yerr=puce_A_2s['V0x_set_average_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0y_set_average_list'],'v',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0y_set_average_list'], yerr=puce_A_2s['V0y_set_average_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$V_0$ $($m$/$s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # # plt.xlim([0,1000])
# # # plt.ylim([0.1,10])
# # plt.plot([1,1000],[1e-4,1e-3],'k--')


# # plt.subplot(122)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['Dx_set_average_list']/puce_A_2s['Taux_fit_estimate_list'])/puce_A_2s['V0y_set_average_list'],'X',linewidth=3, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['D_ech_set_average_list']/puce_A_2s['Taux_fit_estimate_list'])/puce_A_2s['V0y_set_average_list'],'*',linewidth=3, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0x_set_average_list']/puce_A_2s['V0y_set_average_list'],'^',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0x_set_average_list']/puce_A_2s['V0y_set_average_list'], yerr=puce_A_2s['V0x_set_average_list_err']/puce_A_2s['V0y_set_average_list'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list'],'v',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list'], yerr=puce_A_2s['V0y_set_average_list_err']/puce_A_2s['V0y_set_average_list'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$V_x/V_y$ $(1)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # # plt.xlim([0,1000])
# # # plt.ylim([0.1,10])
# # plt.plot([1,10000],[1,100],'k--')















































# # # %% Figure intéressantes

# # plt.figure(14)
# # plt.suptitle(r'Fluctuation dissipation')

# # plt.subplot(121)

# # plt.plot(puce_A_0p5s['Taux_fit_estimate_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(puce_A_1s['Taux_fit_estimate_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'o',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(puce_A_2s['Taux_fit_estimate_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'o',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(puce_A_4s['Taux_fit_estimate_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'o',linewidth=3, color=color_4s,alpha=0.5)

# # plt.plot(puce_A_0p5s['Taux_fit_estimate_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'*',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(puce_A_1s['Taux_fit_estimate_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(puce_A_2s['Taux_fit_estimate_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(puce_A_4s['Taux_fit_estimate_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'*',linewidth=3, color=color_4s,alpha=0.5)

# # plt.xlabel(r'fit $\tau_x$ $($s$)$')
# # plt.ylabel(r'ech $D_x/(V_x^2/\tau_x)$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.xscale('log')
# # # plt.yscale('log')
# # plt.xlim([0,4])
# # plt.ylim([0,4])

# # plt.plot(range(0,5),range(0,5),'k--')


# # plt.subplot(122)

# # plt.plot(puce_A_0p5s['Taux_fit_estimate_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(puce_A_1s['Taux_fit_estimate_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'o',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(puce_A_2s['Taux_fit_estimate_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'o',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(puce_A_4s['Taux_fit_estimate_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'o',linewidth=3, color=color_4s,alpha=0.5)

# # plt.plot(puce_A_0p5s['Taux_fit_estimate_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'*',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(puce_A_1s['Taux_fit_estimate_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(puce_A_2s['Taux_fit_estimate_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(puce_A_4s['Taux_fit_estimate_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'*',linewidth=3, color=color_4s,alpha=0.5)

# # plt.xlabel(r'fit $\tau_x$ $($s$)$')
# # plt.ylabel(r'ech $D_x/(V_x^2/\tau_x)$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.plot(range(0,4),range(0,4),'k--')
# # # %%




# # plt.figure(15)
# # plt.suptitle(r'D mesure renormalisations')

# # plt.subplot(221)
# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0x_set_average_list'][0]/puce_A_0p5s['V0x_set_average_list'][0]*puce_A_0p5s['Taux_fit_estimate_list'][0],'^',linewidth=3, color=color_0p5s,alpha=0.5)
# # # plt.errorbar(puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'], yerr=0*puce_A_0p5s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0x_set_average_list'][0]/puce_A_1s['V0x_set_average_list'][0]*puce_A_1s['Taux_fit_estimate_list'][0],'^',linewidth=3, color=color_1s,alpha=0.5)
# # # plt.errorbar(puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'], yerr=0*puce_A_1s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list'][0]/puce_A_2s['V0x_set_average_list'][0]*puce_A_2s['Taux_fit_estimate_list'][0],'^',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'], yerr=0*puce_A_2s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0x_set_average_list'][0]/puce_A_4s['V0x_set_average_list'][0]*puce_A_4s['Taux_fit_estimate_list'][0],'^',linewidth=3, color=color_4s,alpha=0.5)
# # # plt.errorbar(puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'], yerr=0*puce_A_4s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_4s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'mesure $D_x/(V_x^2/\tau_x)$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # # plt.plot([1,1000],[1e-2,10],'k--')

# # plt.subplot(222)

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']*puce_A_0p5s['Taux_fit_estimate_list'][0],'v',linewidth=3, color=color_0p5s,alpha=0.5)
# # # plt.errorbar(puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'], yerr=0*puce_A_0p5s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Taux_fit_estimate_list'][0],'v',linewidth=3, color=color_1s,alpha=0.5)
# # # plt.errorbar(puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'], yerr=0*puce_A_1s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']*puce_A_2s['Taux_fit_estimate_list'][0],'v',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'], yerr=0*puce_A_2s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']*puce_A_4s['Taux_fit_estimate_list'][0],'v',linewidth=3, color=color_4s,alpha=0.5)
# # # plt.errorbar(puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'], yerr=0*puce_A_4s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_4s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'mesure $D_x/(V_x^2/\tau_x)$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.subplot(223)


# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0x_set_average_list'][0]/puce_A_0p5s['V0x_set_average_list'][0]*puce_A_0p5s['Taux_fit_estimate_list'][0],'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # # plt.errorbar(puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'], yerr=0*puce_A_0p5s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0x_set_average_list'][0]/puce_A_1s['V0x_set_average_list'][0]*puce_A_1s['Taux_fit_estimate_list'][0],'o',linewidth=3, color=color_1s,alpha=0.5)
# # # plt.errorbar(puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'], yerr=0*puce_A_1s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0x_set_average_list'][0]/puce_A_2s['V0x_set_average_list'][0]*puce_A_2s['Taux_fit_estimate_list'][0],'o',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'], yerr=0*puce_A_2s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0x_set_average_list'][0]/puce_A_4s['V0x_set_average_list'][0]*puce_A_4s['Taux_fit_estimate_list'][0],'o',linewidth=3, color=color_4s,alpha=0.5)
# # # plt.errorbar(puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'], yerr=0*puce_A_4s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_4s,alpha=0.5)

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'ech $D_x/(V_x^2/\tau_x^0)$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # plt.subplot(224)

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0y_set_average_list']/puce_A_0p5s['V0y_set_average_list']*puce_A_0p5s['Taux_fit_estimate_list'][0],'*',linewidth=3, color=color_0p5s,alpha=0.5)
# # # plt.errorbar(puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list'], yerr=0*puce_A_0p5s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0y_set_average_list']/puce_A_1s['V0y_set_average_list']*puce_A_1s['Taux_fit_estimate_list'][0],'*',linewidth=3, color=color_1s,alpha=0.5)
# # # plt.errorbar(puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list'], yerr=0*puce_A_1s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0y_set_average_list']/puce_A_2s['V0y_set_average_list']*puce_A_2s['Taux_fit_estimate_list'][0],'*',linewidth=3, color=color_2s,alpha=0.5)
# # # plt.errorbar(puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list'], yerr=0*puce_A_2s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0y_set_average_list']/puce_A_4s['V0y_set_average_list']*puce_A_4s['Taux_fit_estimate_list'][0],'*',linewidth=3, color=color_4s,alpha=0.5)
# # # plt.errorbar(puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list'], yerr=0*puce_A_4s['Dx_fit_set_average_fit_list_err'], xerr=None,linewidth=1, color=color_4s,alpha=0.5)


# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'ech $D_x/(V_y^2/\tau_x^0)$ $(1)$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)


# # plt.figure(16)


# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_alpha_set_average_list']/4/np.pi/np.pi,'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_alpha_set_average_list']/4/np.pi/np.pi, yerr=0*puce_A_0p5s['Dx_fit_set_average_fit_list_err']/4/np.pi/np.pi, xerr=None,linewidth=1, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_alpha_set_average_list']/4/np.pi/np.pi,'o',linewidth=3, color=color_1s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_alpha_set_average_list']/4/np.pi/np.pi, yerr=0*puce_A_1s['Dx_fit_set_average_fit_list_err']/4/np.pi/np.pi, xerr=None,linewidth=1, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_alpha_set_average_list']/4/np.pi/np.pi,'o',linewidth=3, color=color_2s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_alpha_set_average_list']/4/np.pi/np.pi, yerr=0*puce_A_2s['Dx_fit_set_average_fit_list_err']/4/np.pi/np.pi, xerr=None,linewidth=1, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_alpha_set_average_list']/4/np.pi/np.pi,'o',linewidth=3, color=color_4s,alpha=0.5)
# # plt.errorbar(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_alpha_set_average_list']/4/np.pi/np.pi, yerr=0*puce_A_4s['Dx_fit_set_average_fit_list_err']/4/np.pi/np.pi, xerr=None,linewidth=1, color=color_4s,alpha=0.5)

# # plt.plot([1,1000],0.1*np.pi+0*np.array([1,1]),'k--')

# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'measure $D_\alpha$')
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.grid(color='k', linestyle='-', linewidth=0.5)

# # # %

# # plt.figure(12)
# # plt.suptitle(r'Correlation time')

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Taux_fit_estimate_list'],'X',linewidth=3, color=color_4s,alpha=0.5)


# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['Dx_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'o',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['Dx_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'o',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['Dx_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'o',linewidth=3, color=color_4s,alpha=0.5)


# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['V0x_set_average_list']/puce_A_0p5s['V0x_set_average_list'],'*',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['D_ech_set_average_list']/puce_A_1s['V0x_set_average_list']/puce_A_1s['V0x_set_average_list'],'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['D_ech_set_average_list']/puce_A_2s['V0x_set_average_list']/puce_A_2s['V0x_set_average_list'],'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['D_ech_set_average_list']/puce_A_4s['V0x_set_average_list']/puce_A_4s['V0x_set_average_list'],'*',linewidth=3, color=color_4s,alpha=0.5)


# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$\tau$ $($s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # plt.xlim([0,1000])
# # plt.ylim([0.1,10])
# # plt.plot([1,1000],[1,0.1],'k--')
# # # %

# # plt.figure(17)
# # plt.suptitle(r'swimming spead')

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],puce_A_0p5s['V0x_set_average_list'],'v',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],puce_A_1s['V0x_set_average_list'],'v',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],puce_A_2s['V0x_set_average_list'],'v',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],puce_A_4s['V0x_set_average_list'],'v',linewidth=3, color=color_4s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.sqrt(puce_A_0p5s['Dx_set_average_list']/puce_A_0p5s['Taux_fit_estimate_list']),'o',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.sqrt(puce_A_1s['Dx_set_average_list']/puce_A_1s['Taux_fit_estimate_list']),'o',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['Dx_set_average_list']/puce_A_2s['Taux_fit_estimate_list']),'o',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.sqrt(puce_A_4s['Dx_set_average_list']/puce_A_4s['Taux_fit_estimate_list']),'o',linewidth=3, color=color_4s,alpha=0.5)

# # plt.plot(Ratio*Ratio*puce_A_0p5s['A_list']*puce_A_0p5s['A_list'],np.sqrt(puce_A_0p5s['D_ech_set_average_list']/puce_A_0p5s['Taux_fit_estimate_list']),'*',linewidth=3, color=color_0p5s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_1s['A_list']*puce_A_1s['A_list'],np.sqrt(puce_A_1s['D_ech_set_average_list']/puce_A_1s['Taux_fit_estimate_list']),'*',linewidth=3, color=color_1s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_2s['A_list']*puce_A_2s['A_list'],np.sqrt(puce_A_2s['D_ech_set_average_list']/puce_A_2s['Taux_fit_estimate_list']),'*',linewidth=3, color=color_2s,alpha=0.5)
# # plt.plot(Ratio*Ratio*puce_A_4s['A_list']*puce_A_4s['A_list'],np.sqrt(puce_A_4s['D_ech_set_average_list']/puce_A_4s['Taux_fit_estimate_list']),'*',linewidth=3, color=color_4s,alpha=0.5)


# # plt.xlabel(r'$P^2$ $($mbar$^2)$')
# # plt.ylabel(r'$V_0$ $($m$/$s$)$')

# # plt.grid(color='k', linestyle='-', linewidth=0.5)
# # plt.xscale('log')
# # plt.yscale('log')
# # # plt.xlim([0,1000])
# # # plt.ylim([0.1,10])
# # plt.plot([1,1000],[1e-4,1e-3],'k--')