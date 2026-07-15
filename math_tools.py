from scipy.fftpack import *
import numpy as np 
import addons
import pandas as pd 
import os
import helpers
import random
import math
import math_tools
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation

def filter_signal(data,fs,lowcut,highcut,order=3):
    filtered_signal = {}
    for i in range(len(data)):
        try:
            filtered_signal[i] = addons.butter_bandpass_filter(data[i]-np.mean(data[i]),lowcut,highcut,fs,order)
        except:
            filtered_signal[i] = data[i]

    return filtered_signal

def filter_signal_2(data,fs,lowcut,highcut,order=3):
      return addons.butter_bandpass_filter(data-np.mean(data),lowcut,highcut,fs,order)



def hilbert_transform(signal):
    '''
    N : fft length
    M : number of elements to zero out
    U : DFT of signal
    V: IDFT of H(U) 
    '''

    N = len(signal)
    #take the forward Fourier transform
    U = fft(signal)
    M = N - N//2 - 1
    #Zero out negative frequency components
    U[N//2+1:] = [0] * M 
    #double fft energy except #DC0
    U[1:N//2] = 2 * U[1:N//2]
    #take inverse of Fourier transform
    v = ifft(U)
    return v 

def analyse_synchrony(num_neurons,name,bin_width=3,t_r = 2,dt=0.01):
    analysis_interval_start = addons.analysis_dict["analysis_start"]
    analysis_interval_end = addons.analysis_dict["analysis_end"]
    analysis_interval_start_s = addons.analysis_dict["synchrony_start"]
    analysis_interval_end_s = addons.analysis_dict["synchrony_end"]
    
    
    analysis_length = analysis_interval_end - analysis_interval_start
    analysis_length_s = analysis_interval_end_s - analysis_interval_start_s
    if analysis_length - analysis_length_s < 0:
        print('There is a problem. Synchrony measurement range must be smaller than the other')
        return 

    sd_names, node_ids, data = helpers.__load_spike_times(name,"spike_recorder",analysis_interval_start_s, analysis_interval_end_s)

    data_s = {}

    for i in data:
        low = np.searchsorted(data[i]["time_ms"],v=analysis_interval_start_s,side="left")
        high = np.searchsorted(data[i]["time_ms"],v=analysis_interval_end_s,side='right')
        data_s[i] = data[i][low:high]


    synchrony_pd = []
    irregularity = []
    irregularity_pdf = {}
    times_s = {}
    for i, n in enumerate(sd_names):

        #Computing synchrony
        neurons = np.unique(data_s[i]["sender"])
        random.shuffle(neurons)
        chosen_ones = neurons[1:1000]
        indices = []
        for indx in chosen_ones:
            indices = np.append(indices,np.where(data_s[i]["sender"]==indx))
        indices = np.array(indices,dtype=int)
        times_s[i] = data_s[i][indices]["time_ms"]
        counts, bins = np.histogram(times_s[i], bins=int(analysis_length_s/bin_width))
        if np.mean(counts) == 0:
            synchrony_pd = np.append(synchrony_pd,0)
        else:
            synchrony_pd = np.append(synchrony_pd,np.var(counts)/np.mean(counts))

    for i,n in enumerate(sd_names):
        single_irregularity = []
        used_senders = []
        single_lvr = 0
        lvr = []
        for senders in data[i]["sender"]:
            individual_neurons = []
            times = []
            isi = []
            count = 0
            if senders not in used_senders:
                used_senders = np.append(used_senders,senders)
                individual_neurons = np.append(individual_neurons,np.where(data[i]["sender"]==senders))
                for index in individual_neurons:
                    times = np.append(times,data[i][int(index)]["time_ms"])

                if len(times)>4:
                    for j in range(len(times)-1):
                        isi = np.append(isi,times[j+1]-times[j])
                    mean =np.mean(isi)
                    var = np.sqrt(np.var(isi))
                    single_irregularity = np.append(single_irregularity,np.float128(var/mean))
                    count = count + 1    
                if count >= 1000:
                    break

        if np.mean(single_irregularity)==None:
            irregularity = np.append(irregularity,0)
        else:
            irregularity = np.append(irregularity,np.mean(single_irregularity))
            irregularity_pdf[i] = single_irregularity
 
    return synchrony_pd, irregularity, irregularity_pdf, times_s 

def plot_synchrony(synchrony_pd, synchrony_chi, irregularity_pdf, name):

    a = plt.figure(figsize=(18,18))

#####################################################################################################################
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    fs = 18
    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

    plt.subplot(3, 1, 1)
    plt.barh(pops[::-1], synchrony_pd[::-1], color = colours[::-1])
    plt.xlabel('synchrony', fontsize = fs)
    plt.grid(alpha = 0.5)


###########################################################################################################################
    plt.subplot(3,1,2)
    test = []
    colours_box = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    pops_box = pops
    j = 0
    label_pos = list(range(len(pops_box),0,-1))
    for i in np.arange(len(pops)):
        if len(irregularity_pdf[i]) == 0: 
            del pops_box[i-j]
            j = j+1
        else:
            test.append(irregularity_pdf[i])
    bp = plt.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True,orientation='horizontal')
    label_pos = list(range(len(pops_box), 0, -1))
    if len(pops) != len(pops_box):
        colours_box = ['#3288bd', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    for i in np.arange(len(pops_box)):
        boxX = []
        boxY = []
        box = bp["boxes"][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colours_box[-i])
        plt.gca().add_patch(boxPolygon)
    plt.xlabel('irregularity', fontsize = fs)
    plt.yticks(label_pos, pops_box, fontsize=fs)
    plt.grid(alpha = 0.5)

############################################################################################################
    plt.subplot(3,1,3)
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
    colours_pdf = list(reversed(colours))
    pops_pdf = list(reversed(pops))
    irregularity_total = []
    for i in irregularity_pdf:
        data, bins= np.histogram(irregularity_pdf[i],density=True,bins=50)
        irregularity_total = np.append(irregularity_total,irregularity_pdf[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops_pdf[i], color = colours_pdf[i])

    data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
    plt.legend()
    plt.ylabel('P (irregularity)', fontsize = fs)
    plt.xlabel('CV ISI', fontsize = fs)
    plt.grid(alpha = 0.5)
    plt.xlim(0,1.8)
    plt.ylim(0,3.6)