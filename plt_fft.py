from scipy.sparse import dok_matrix 
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy import signal
from scipy.signal import butter, lfilter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
import addons
import numpy as np
import pandas as pd 
import os
import helpers
import random
import math
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation


def filter_signal(data,fs,lowcut,highcut,order=3):
    filtered_signal = {}
    for i in data:
        filtered_signal[i] = addons.butter_bandpass_filter(data[i][addons.analysis_dict["analysis_start"]:addons.analysis_dict["analysis_end"]]-np.mean(data[i][addons.analysis_dict["analysis_start"]:addons.analysis_dict["analysis_end"]]),lowcut,highcut,fs,order)

    return filtered_signal

def compute_FFT(signal_data,name_,rate,freq_sample= 0.001,freq_sample_welsh = 1000,lim_y = 7000, lim_x = 200, low_log = 10, high_log =90,fit=False,fit_freq_start = 4.0, fit_freq_end = 14.0,test_p0 = [30,10,5],welsh_fit = 'alpha',signal_xmin=500,signal_xmax=900,min_x=1,save=True):

    analysis_interval_start = addons.analysis_dict["analysis_start"]
    analysis_interval_end = addons.analysis_dict["analysis_end"]
    FFT_Results = {}
    Welsh_Freqs = {}
    Welsh_Powers = {}
    if fit:
        Fit_FFT = {}
        Fit_Welsh = {}
        mean_freq = []
        mean_welsh = []
        amplitude_freq = []
        amplitude_welsh = []
        sigma_freq = []
        sigma_welsh = []
        def gaus(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))


    for i in signal_data:
        FFT_Results[i] = fft(signal_data[i][analysis_interval_start:analysis_interval_end]-np.mean(signal_data[i][analysis_interval_start:analysis_interval_end]))
        Welsh_Freqs[i], Welsh_Powers[i]  = signal.welch(signal_data[i][analysis_interval_start:analysis_interval_end]-np.mean(signal_data[i][analysis_interval_start:analysis_interval_end]),fs=freq_sample_welsh)

    #Calcular los valores de frequencia correspondientes
    #freq = fftfreq(len(signal_data[i][analysis_interval_start:analysis_interval_end]),d=freq_sample) * 1000
    freq = fftfreq(len(signal_data[i][analysis_interval_start:analysis_interval_end]),d=freq_sample)
    index_start = int(np.where(freq==fit_freq_start)[0][0])
    index_end = int(np.where(freq==fit_freq_end)[0][0])
    

    plt.figure(figsize=(25, 15))
    colors = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    pops = ["L23E","L23I","L4E","L4I","L5E","L5I","L6E","L6I"]
    # Graficar la amplitud en función de la frecuencia

    plt.subplot(2, 1, 1)
    j= 0
    for i in signal_data:
        plt.plot(signal_data[i][analysis_interval_start:analysis_interval_end+700], c = colors[j], label = pops[i])
        j=j+1
    plt.xlabel('Time (ms)')
    plt.ylabel('Signal')
    plt.title('Signal, background rate = ' +str(rate))
    plt.xlim(signal_xmin,signal_xmax)    
    plt.grid(True)
    #plt.legend(loc= 'best')
    plt.legend(loc= 'upper left')
    plt.subplot(2, 3, 4)
    j= 0

    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            Fit_FFT[i], __ = curve_fit(gaus,freq[index_start:index_end],np.abs(FFT_Results[i][index_start:index_end]),p0 = test_p0)
            mean_freq = np.append(mean_freq,Fit_FFT[i][1])
            amplitude_freq = np.append(amplitude_freq,Fit_FFT[i][0])
            sigma_freq = np.append(sigma_freq,Fit_FFT[i][2])

            plt.plot(freq[index_start:index_end],gaus(freq[index_start:index_end],*Fit_FFT[i]),'--', c = colors[j])
        plt.plot(freq[:indx], np.abs(FFT_Results[i])[:indx],c = colors[j], label = pops[i])
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.figtext(0.32,0.50,"Fast Fourier Transform", va="center", ha="center", size=22)
    plt.grid(True)
    plt.xlim(min_x,lim_x)
    plt.ylim(0,lim_y)
    #plt.legend(loc= 'best')

    plt.subplot(2, 3, 5)
    j= 0
    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            plt.plot(freq[index_start:index_end],20 * np.log10(gaus(freq[index_start:index_end],*Fit_FFT[i])),'--', c = colors[j])
        plt.plot(freq[:indx], 20 * np.log10(np.abs(FFT_Results[i])[:indx]),c = colors[j], label = pops[i])
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    #plt.title('Voltage (minus the mean) FFT')
    plt.grid(True)
    plt.xlim(min_x,lim_x)
    plt.ylim(low_log,high_log)
    #plt.legend(loc= 'best')

    plt.subplot(2, 3, 6)
    j= 0

    #plt.ylim([0.5e-3, 1])
    for i in Welsh_Freqs:
        if fit:
            if welsh_fit == 'alpha':
                i_start = int(np.where((Welsh_Freqs[i] < 4) & (Welsh_Freqs[i] > 0.0))[0][0])
                i_end = int(np.where((Welsh_Freqs[i]<22.0) & (Welsh_Freqs[i] >18.0))[0][0])
                p0 = [0.01,10,5]

            if welsh_fit == 'gamma':
                i_start = int(np.where((Welsh_Freqs[i] <60 ) & (Welsh_Freqs[i] > 50.0))[0][0])
                i_end = int(np.where((Welsh_Freqs[i]<120.0) & (Welsh_Freqs[i] >110.0))[0][0])
                p0 = [0.01,80,5]
        if fit:
            Fit_Welsh[i], __ = curve_fit(gaus,Welsh_Freqs[i][i_start:i_end],Welsh_Powers[i][i_start:i_end],p0 = p0)
            mean_welsh = np.append(mean_welsh,Fit_Welsh[i][1])
            amplitude_welsh = np.append(amplitude_welsh,Fit_Welsh[i][0])
            sigma_welsh = np.append(sigma_welsh,Fit_Welsh[i][2])
            plt.plot(Welsh_Freqs[i][i_start:i_end],gaus(Welsh_Freqs[i][i_start:i_end],*Fit_Welsh[i]),'--', c = colors[j])
        plt.plot(Welsh_Freqs[i], Welsh_Powers[i],c = colors[j], label = pops[i])
        j=j+1
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD $[V^2/Hz]$')
    plt.title('Welsh PSD',fontsize=22)
    plt.grid(True)
    
  
    plt.xlim(min_x,lim_x)
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.show()

    if save:
        mean_final = np.mean( np.array([mean_freq,mean_welsh]),axis= 0)
        amplitude_final = np.mean( np.array([amplitude_freq,amplitude_welsh]),axis= 0)
        sigma_final = np.mean( np.array([sigma_freq,sigma_welsh]),axis= 0)
        pops = [0,1,2,3,4,5,6,7]
        np.savetxt(name_, np.c_[pops,mean_final, amplitude_final, sigma_final], fmt = '%.2f', header = 'Pops mean_freq amplitude sigma')

    return FFT_Results, freq, Welsh_Freqs, Welsh_Powers, indx

bg_rate = np.linspace(4,12,int((12-4)/0.25))
list_dirs = os.listdir("data_background_rate")
list_dirs = sorted(list_dirs)


pop_activity = {}
names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]

type = 'normal' #normal, alpha, gamma

for i, dir in enumerate(list_dirs):
    name = "data_background_rate/" + str(dir)
    neuron_id = np.loadtxt((os.path.join(name,"population_nodeids.dat")),dtype=int)
    num_neurons = []
    for l in range(len(neuron_id)):
        num_neurons = np.append(num_neurons,int(neuron_id[l][1]-neuron_id[l][0]+1))
    num_neurons = num_neurons.astype(int)
    pop_activity[bg_rate[i]] = {}
    for j in range(len(num_neurons)):
        pop_activity[bg_rate[i]][j] = np.loadtxt(name+"/measurements/pop_activities/pop_activity_"+str(j)+".dat")
        if type == 'alpha':
            lowcut_alpha = 8
            highcut_alpha =12
            pop_activity[bg_rate[i]] = filter_signal(pop_activity[bg_rate[i]],lowcut=lowcut_alpha,highcut=highcut_alpha,fs=1000)
        if type == 'gamma':
            lowcut_gamma = 50
            highcut_gamma = 95
            pop_activity[bg_rate[i]] = filter_signal(pop_activity[bg_rate[i]],lowcut=lowcut_gamma,highcut=highcut_gamma,fs=1000)



name = 'data_background_rate/'+str(i)  +'/results/fft.dat'
if type == 'alpha':
    Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[bg_rate[16]],name_=name,rate=bg_rate[16],lim_y=100000,lim_x=100,high_log=100,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)
elif type == 'gamma':
    Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[bg_rate[16]],name_=name,rate=bg_rate[16],lim_y=400000,lim_x=150,high_log=130,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)
else:
    Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[bg_rate[16]],name_=name,rate=bg_rate[16],lim_y=200000,lim_x=150,high_log=120,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)


bg_rate_ini = 8.13
fs = 18
pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']

fig, ax = plt.subplots(figsize=(25,25))

axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=bg_rate[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    plt.close()
    if type == 'alpha':
        Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[slider.val],name_=name,rate=slider.val,lim_y=100000,lim_x=100,high_log=100,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)
    elif type == 'gamma':
        Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[slider.val],name_=name,rate=slider.val,lim_y=400000,lim_x=150,high_log=130,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)
    else:
        Fourier_signal, FFT_frequencies, Welsh_frequencies, Welsh_signal, FFT_index  = compute_FFT(pop_activity[slider.val],name_=name,rate=slider.val,lim_y=200000,lim_x=150,high_log=120,low_log=0,freq_sample=0.001,freq_sample_welsh=1000,signal_xmax=1500,save=False)
    fig.canvas.draw_idle()

slider.on_changed(update)

plt.show()

