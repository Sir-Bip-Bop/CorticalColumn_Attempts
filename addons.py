import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nest 
import helpers
from matplotlib.patches import Polygon
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy.fft import fft
from scipy.fft import fftfreq
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.signal import freqz
from scipy.optimize import curve_fit
import scienceplots
from scipy import signal
import random
plt.style.use(['science'])


analysis_dict = {
    "analysis_start": 500,
    "analysis_end": 5500,
    "name": "bg_8_delay_2.5/", 
    "synchrony_start": 500,
    "synchrony_end": 60500,
    "convolve_bin_size": 0.2,
    "bin_size": 3,
    }

def number_synapses(net_pops,number = 50):
    data_synapses = {}
    j = 0

    population_limits = np.loadtxt("data_og/population_nodeids.dat")
    ex_populations = np.array(population_limits[::2],dtype=int)
    ranges = []

    for i in range(len(ex_populations)):
        ranges =  np.append(ranges,range(ex_populations[i][0],ex_populations[i][1]))

    for pop in net_pops:
        column_list = []
        counter = 0
        data_synapses[j] = {}
        data_frame = {}
        print("\r pop: %d " %(j), end = '\n',flush = True)
        for neuron in pop:
            connections = nest.GetConnections(target=neuron)
            conn_vals = connections.get(["target","source"])
            column_list = np.append(column_list,conn_vals["target"][0])
            print("\r neuron: %d " % (conn_vals["target"][0]),end = '',flush=True)
            ex_counts = 0
            in_counts = 0
            for i in range(len(conn_vals["source"])):
                if conn_vals["source"][i] in ranges:
                    ex_counts = ex_counts +1
                else:
                    in_counts = in_counts + 1
            data_frame[conn_vals["target"][0]] = [ex_counts,in_counts]
            counter = counter +1 
            if counter >= number:
                break
        dataframe = pd.DataFrame(data=data_frame, columns= column_list, index=["excitatory","inhibitory"])
        dataframe.to_csv("synapses_data/pop"+ str(j)+".txt")
        data_synapses[j] = dataframe
        j = j +1
    
    return data_synapses

def gather_metadata(path, name):
    """Reads names and ids of spike recorders and first and last ids of
    neurons in each population.

    If the simulation was run on several threads or MPI-processes, one name per
    spike recorder per MPI-process/thread is extracted.

    Parameters
    ------------
    path
        Path where the spike recorder files are stored.
    name
        Name of the spike recorder, typically ``spike_recorder``.

    Returns
    -------
    sd_files
        Names of all files written by spike recorders.
    sd_names
        Names of all spike recorders.
    node_ids
        Lowest and highest id of nodes in each population.

    """
    # load filenames
    sd_files = []
    sd_names = []
    for fn in sorted(os.listdir(path)):
        if fn.startswith(name):
            sd_files.append(fn)
            fnsplit = "-".join(fn.split("-")[:-1])
            if fnsplit not in sd_names:
                sd_names.append(fnsplit)
    
    #load node IDs
    node_idfile = open(path + "population_nodeids.dat","r")
    node_ids = []
    for node_id in node_idfile:
        node_ids.append(node_id.split())
    node_ids = np.array(node_ids, dtype="i4")
    return sd_files, sd_names, node_ids

def load_voltage(path, name):
    sd_files, sd_names, node_ids = gather_metadata(path,name)
    data = {}
    dtype = {"names": ("sender","time_ms","V_m"), "formats": ("i4","f8","f8")}

    for i,name in enumerate(sd_names):
        data_i_raw = np.array([[]], dtype=dtype)
        for j, f in enumerate(sd_files):
            if name in f:
                ld = np.loadtxt(os.path.join(path,f), skiprows=3,dtype=dtype)
                data_i_raw = np.append(data_i_raw, ld)

        data_i_raw = np.sort(data_i_raw, order = "time_ms")
        data[i] = data_i_raw

    return data

def load_current(path,name):
    sd_files, sd_names, node_ids = gather_metadata(path,name)
    data = {}
    dtype = {"names": ("sender","time_ms","I_syn"), "formats": ("i4","f8","f8")}

    for i,name in enumerate(sd_names):
        data_i_raw = np.array([[]],dtype=dtype)
        for j,f in enumerate(sd_files):
            if name in f:
                ld = np.loadtxt(os.path.join(path,f),skiprows=3,dtype=dtype)
                data_i_raw = np.append(data_i_raw,ld)
        data_i_raw = np.sort(data_i_raw,order = "time_ms")
        data[i] = data_i_raw

    return data 

def split_voltage(data,num_neurons):
    data_pop = np.zeros((num_neurons,len(data["time_ms"][0::num_neurons])))
    for i in range(num_neurons):
        data_pop[i][:] = data["V_m"][i::num_neurons]
    return data_pop

def split_current(data,num_neurons):
    data_pop = np.zeros((num_neurons,len(data["time_ms"][0::num_neurons])))
    for i in range(num_neurons):
        data_pop[i][:] = data["I_syn"][i::num_neurons]
    return data_pop

def get_time(data,num_neurons):
    return data[0]["time_ms"][0::num_neurons]


def analyse_synchrony(bin_width=3,t_r = 2):
    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
    analysis_interval_start_s = analysis_dict["synchrony_start"]
    analysis_interval_end_s = analysis_dict["synchrony_end"]
    

    analysis_length = analysis_interval_end - analysis_interval_start
    analysis_length_s = analysis_interval_end_s - analysis_interval_start_s

    if analysis_length_s - analysis_length < 0:
        print('There is a problem. Synchrony measurement range must be larger than the other')
        return 

    sd_names_s, node_ids, data_s = helpers.__load_spike_times("data_og/","spike_recorder",analysis_interval_start_s, analysis_interval_end_s)
    
    data = {}

    for i in data_s:
        low = np.searchsorted(data_s[i]["time_ms"],v=analysis_interval_start,side="left")
        high = np.searchsorted(data_s[i]["time_ms"],v=analysis_interval_end,side='right')
        data[i] = data_s[i][low:high]


    synchrony_pd = []
    irregularity = []

    super_lvr = []
    irregularity_pdf = {}
    lvr_pdf = {}

    times_s = {}

    for i, n in enumerate(sd_names_s):

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
        synchrony_pd = np.append(synchrony_pd,np.var(counts)/np.mean(counts))

        #Computing irregularity and LvR
        single_irregularity = []
        used_senders = []
        single_lvr = 0
        lvr = []
    
    for i,n in enumerate(sd_names_s):
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
                    for j in range(len(isi)-1):
                        single_lvr = single_lvr +  (1 - 4*isi[j]*isi[j+1] / (isi[j]+isi[j+1])**2) * (1 + 4 * t_r / (isi[j] + isi[j+1]))

                    neuron_rate = len(times) / analysis_length * 1000

                    single_lvr = 3 / (len(isi)-1) * single_lvr
                    lvr = np.append(lvr, single_lvr)
                    mean =np.mean(isi)
                    var = np.sqrt(np.var(isi))
                    single_irregularity = np.append(single_irregularity,np.float128(var/mean))
                    count = count + 1

        super_lvr = np.append(super_lvr, np.mean(lvr))
        lvr_pdf[i] = lvr
        irregularity = np.append(irregularity,np.mean(single_irregularity))
        irregularity_pdf[i] = single_irregularity
        if count >= 1000:
            break

    super_lvr = super_lvr[~np.isnan(super_lvr)]
    irregularity = irregularity[~np.isnan(irregularity)] 

    return synchrony_pd, irregularity, irregularity_pdf, super_lvr, lvr_pdf, times_s


def analyse_firing_rates():
    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
    analysis_length = analysis_interval_end - analysis_interval_start
    
    sd_names, node_ids, data = helpers.__load_spike_times("data_og/","spike_recorder",analysis_interval_start, analysis_interval_end)

    spike_rates = {}
    super_times = []

    for i, n in enumerate(sd_names):
        single_spike_rates = []
        used_senders = []
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
                if len(times)>2:
                    neuron_rate = len(times) / analysis_length * 1000
                    single_spike_rates = np.append(single_spike_rates,neuron_rate)
                    count = count + 1
        spike_rates[i] = single_spike_rates
        if count >= 1000:
            break
    return spike_rates

def prepare_data(data_pop,ex_current_pop,in_current_pop):
    sd_names, node_ids, data = helpers.__load_spike_times("data_og/","spike_recorder",analysis_dict["analysis_start"], analysis_dict["analysis_end"])

    times = {}
    data_voltages = { }
    data_excitatory = {}
    data_inhibitory = {}
    bins = {}

    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]



    for i in range(len(data_pop)):
        random.shuffle(data_pop[i])
        random.shuffle(ex_current_pop[i])
        random.shuffle(in_current_pop[i])
        data_voltages[names[i]] =  np.mean(data_pop[i][0:1000],axis=0)
        data_excitatory[names[i]] = np.mean(ex_current_pop[i][0:1000],axis=0)
        data_inhibitory[names[i]] = np.mean(in_current_pop[i][0:1000],axis=0)
        neurons = np.unique(data[i]["sender"]) 
        random.shuffle(neurons)
        chosen_ones = neurons[1:1000]
        indices = []
        for indx in chosen_ones:
            indices = np.append(indices,np.where(data[i]["sender"]==indx))
        indices = np.array(indices,dtype=int)
        times_help = data[i][indices]["time_ms"] 
        times[names[i]], bins[names[i]] = np.histogram(data[i][indices]["time_ms"], bins = int(analysis_dict["analysis_end"]-analysis_dict["analysis_start"]/analysis_dict["bin_size"]))

    return data_voltages, data_excitatory, data_inhibitory, times, times_help


def plot_correlations(data_voltages,data_excitatory,data_inhibitory,pop_activity,times,save_data=True):
    connection_p =np.array([[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.0, 0.0076, 0.0],
            [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.0, 0.0042, 0.0],
            [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.0],
            [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.0, 0.1057, 0.0],
            [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.0],
            [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.0],
            [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
            [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443],
        ]
    )
    names = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    dataframe = pd.DataFrame(data=data_voltages, columns= names)
    matrix = dataframe.corr(method='pearson')

    dataframe_ex = pd.DataFrame(data=data_excitatory, columns= names)
    matrix_ex = dataframe_ex.corr(method='pearson')

    dataframe_in = pd.DataFrame(data=data_inhibitory, columns= names)
    matrix_in = dataframe_in.corr(method='pearson')
    dataframe_activity = pd.DataFrame(data=pop_activity, columns= names)
    matrix_activity = dataframe_activity.corr(method='pearson')

    dataframe_times = pd.DataFrame(data=times, columns= names)
    matrix_times = dataframe_times.corr(method='pearson')

    variables = []
    for i in matrix.columns:
        variables.append(i)

    variablest = []
    for i in matrix_times.columns:
        variablest.append(i)

    variables_ex = []
    for i in matrix_ex.columns:
        variables_ex.append(i)

    variables_activity = []
    for i in matrix_activity.columns:
        variables_activity.append(i)

    variables_in = []
    for i in matrix_in.columns:
        variables_in.append(i)

    plt.figure(figsize=(25,10))

    # Adding labels to the matrix
    plt.subplot(2, 3, 1)
    plt.imshow(matrix, cmap='Blues')
    plt.colorbar()
    plt.title('Mean Voltage Correlation')
    plt.xticks(range(len(matrix)), variables, rotation=45, ha='right')
    plt.yticks(range(len(matrix)), variables)

    plt.subplot(2, 3, 2)    
    plt.imshow(matrix_ex, cmap='Blues')
    plt.colorbar()
    plt.title('Excitatory Current Correlation')
    plt.xticks(range(len(matrix_ex)), variables_ex, rotation=45, ha='right')
    plt.yticks(range(len(matrix_ex)), variables_ex)

    plt.subplot(2, 3, 3)
    plt.imshow(matrix_in, cmap='Blues')
    plt.colorbar()
    plt.title('Inhibitory Current Correlation')
    plt.xticks(range(len(matrix_in)), variables_in, rotation=45, ha='right')
    plt.yticks(range(len(matrix_in)), variables_in)

    plt.subplot(2, 3, 4)
    plt.imshow(matrix_activity, cmap='Blues')
    plt.colorbar()
    plt.title('Pop Activity Correlation')
    plt.xticks(range(len(matrix_activity)), variables_activity, rotation=45, ha='right')
    plt.yticks(range(len(matrix_activity)), variables_activity)    
    
    plt.subplot(2, 3, 5)
    plt.imshow(matrix_times, cmap='Blues')
    plt.colorbar()
    plt.title('Spike Trains Correlation')
    plt.xticks(range(len(matrix_times)), variablest, rotation=45, ha='right')
    plt.yticks(range(len(matrix_times)), variablest)

    plt.subplot(2, 3, 6)
    plt.imshow(connection_p, cmap = 'Greens')
    plt.xticks(ticks=[0,1,2,3,4,5,6,7],labels=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"])
    plt.yticks(ticks=[0,1,2,3,4,5,6,7],labels=["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"])
    plt.title('Connection probabilities')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    
    if save_data:
        matrix.to_csv(analysis_dict["name"]+'voltage_correlation.dat', sep=' ')
        matrix_times.to_csv(analysis_dict["name"] + 'spiketimes_correlation.dat', sep=' ')
        matrix_ex.to_csv(analysis_dict["name"] + 'ex_current_correlation.dat', sep=' ')
        matrix_in.to_csv(analysis_dict["name"] + 'in_currents_correlation.dat', sep=' ')
        matrix_activity.to_csv(analysis_dict["name"] + 'activity_correlation.dat', sep=' ')

def plot_cross_correlation(signal_1,signal_packet,signal_name,time_lag = 50, corr_start = 2000, corr_end = 2200):
    fig = plt.figure(figsize=(20,11))
    ax = fig.add_subplot(121,label='1')
    ax2 = fig.add_subplot(122, label = "2")
    ax.plot(signal_1[corr_start:corr_end], color = "blue", label = "Original -"+signal_name)
    for i, n in enumerate(signal_packet):
        if str(n) != signal_name:
            corr = signal.correlate(signal_1[corr_start:corr_end],signal_packet[str(n)][corr_start-time_lag:corr_end+time_lag],mode='valid')
            lags = signal.correlation_lags(len(signal_1[corr_start:corr_end]), len(signal_packet[str(n)][corr_start-time_lag:corr_end+time_lag]),mode='valid')
            corr /= np.max(corr)
            ax.plot(signal_packet[str(n)][corr_start:corr_end],label = str(n))
            ax2.plot(lags, corr,label=str(n))

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.legend()
    ax.grid()
    ax2.set_xlabel("Discretised time lag")
    ax2.set_ylabel("Normalised correlation")
    ax2.grid()




def plot_synchrony(synchrony_pd, irregularity, irregularity_pdf, lvr, lvr_pdf):
    plt.figure(figsize=(18,18))
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    bar_labels = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']

    plt.subplot(3, 2, 1)

    plt.barh(pops, synchrony_pd, color = bar_labels)
    plt.ylabel('Populations')
    plt.title('Synchrony')
    plt.xlabel('Synchrony')

    plt.subplot(3, 2, 2)
    plt.barh(pops, synchrony_pd, color = bar_labels)
    plt.ylabel('Populations')
    plt.title('Placeholder')
    plt.xlabel('Synchrony')

    plt.subplot(3,2,3)
    plt.barh(pops, irregularity, color = bar_labels)
    plt.ylabel('Populations')
    plt.title('Irregularity')
    plt.xlabel('Irregulatiry')

    plt.subplot(3,2,4)
    irregularity_total = []
    for i in irregularity_pdf:
        data, bins= np.histogram(irregularity_pdf[i],density=True,bins=50)
        irregularity_total = np.append(irregularity_total,irregularity_pdf[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = bar_labels[i])

    data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
    plt.grid()
    plt.legend()
    plt.ylabel('Normalised Counts')
    plt.title('P (irregularity)')
    plt.xlabel('CV ISI')

    plt.subplot(3,2,5)
    plt.barh(pops,lvr, color = bar_labels)
    plt.ylabel('Populations')
    plt.title('LvR')
    plt.xlabel('LvR')

    plt.subplot(3,2,6)
    lvr_total = []
    for i in lvr_pdf:
        data, bins= np.histogram(lvr_pdf[i],density=True)
        lvt_total = np.append(lvr_total,lvr_pdf[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = bar_labels[i])

    data_t, bins_s = np.histogram(lvr_total,density=True)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
    plt.grid()
    plt.legend()
    plt.ylabel('Normalised Counts')
    plt.title('P (LvR)')
    plt.xlabel('LvR')
    plt.show()

def plot_firing_rates(spike_rates):
    plt.figure(figsize=(12,7))
    pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
    bar_labels = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    bar_colors = ['tab:red', 'tab:blue', 'tab:red', 'tab:orange']
    plt.subplot(1,2,1)
    for i in spike_rates:
        plt.hist(spike_rates[i],color=bar_labels[i],alpha = 0.3 + i*0.1, label = pops[i])
    plt.grid()
    plt.legend()
    plt.ylabel('Counts')
    plt.title('Firing rate histogram')
    plt.xlabel('Firing rate (spikes/s)')

    plt.subplot(1,2,2)
    spike_total = []
    for i in spike_rates:
        data, bins= np.histogram(spike_rates[i],density=True)
        spike_total = np.append(spike_total,spike_rates[i])
        plt.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops[i], color = bar_labels[i])

    data_t, bins_s = np.histogram(spike_total,density=True)
    plt.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
    plt.grid()
    plt.legend()
    plt.ylabel('Normalised Counts')
    plt.title('P (rate)')
    plt.xlabel('Firing rate (spikes/s)')
    plt.show()


def compute_FFT(signal_data,freq_sample= 0.001,freq_sample_welsh = 1000,lim_y = 7000, lim_x = 200, low_log = 10, high_log =90,fit=False,fit_freq_start = 4.0, fit_freq_end = 14.0,test_p0 = [30,10,5],welsh_fit = 'alpha',signal_xmin=500,signal_xmax=700,save=True,name_='freq.dat'):

    analysis_interval_start = analysis_dict["analysis_start"]
    analysis_interval_end = analysis_dict["analysis_end"]
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
    freq = fftfreq(len(signal_data[i][analysis_interval_start:analysis_interval_end]),d=freq_sample)
    index_start = int(np.where(freq==fit_freq_start)[0][0])
    index_end = int(np.where(freq==fit_freq_end)[0][0])


    plt.figure(figsize=(15, 5))
    colors = ['darkred', 'red', 'blue', 'aqua', 'green', 'lime', 'orange', 'moccasin']
    # Graficar la amplitud en función de la frecuencia

    plt.subplot(1, 4, 1)
    j= 0
    for i in signal_data:
        plt.plot(signal_data[i][analysis_interval_start:analysis_interval_end], c = colors[j], label = i)
        j=j+1
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage')
    plt.title('Signal')
    plt.xlim(signal_xmin,signal_xmax)    
    plt.grid(True)
    plt.legend(loc= 'best')

    plt.subplot(1, 4, 2)
    j= 0

    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            Fit_FFT[i], __ = curve_fit(gaus,freq[index_start:index_end],np.abs(FFT_Results[i][index_start:index_end]),p0 = test_p0)
            mean_freq = np.append(mean_freq,Fit_FFT[i][1])
            amplitude_freq = np.append(amplitude_freq,Fit_FFT[i][0])
            sigma_freq = np.append(sigma_freq,Fit_FFT[i][2])

            plt.plot(freq[index_start:index_end],gaus(freq[index_start:index_end],*Fit_FFT[i]),'--', c = colors[j])
        plt.plot(freq[:indx], np.abs(FFT_Results[i])[:indx],c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Voltage (minus the mean) FFT')
    plt.grid(True)
    plt.xlim(1,lim_x)
    plt.ylim(0,lim_y)
    plt.legend(loc= 'best')

    plt.subplot(1, 4, 3)
    j= 0
    indx = int(len(signal_data[i][analysis_interval_start:analysis_interval_end])/2)
    for i in FFT_Results:
        if fit:
            plt.plot(freq[index_start:index_end],20 * np.log10(gaus(freq[index_start:index_end],*Fit_FFT[i])),'--', c = colors[j])
        plt.plot(freq[:indx], 20 * np.log10(np.abs(FFT_Results[i])[:indx]),c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude (dB)')
    plt.title('Voltage (minus the mean) FFT')
    plt.grid(True)
    plt.xlim(1,lim_x)
    plt.ylim(low_log,high_log)
    plt.legend(loc= 'best')

    plt.subplot(1, 4, 4)
    j= 0

    #plt.ylim([0.5e-3, 1])
    for i in Welsh_Freqs:
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
        plt.plot(Welsh_Freqs[i], Welsh_Powers[i],c = colors[j], label = i)
        j=j+1
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'PSD $[V^2/Hz]$')
    plt.title('Voltage (minus the mean) Welsh')
    plt.grid(True)
    plt.legend(loc= 'best')
    plt.tight_layout()
    plt.xlim(1,lim_x)
    plt.yscale('log')
    plt.show()

    if save:
        mean_final = np.mean( np.array([mean_freq,mean_welsh]),axis= 0)
        amplitude_final = np.mean( np.array([amplitude_freq,amplitude_welsh]),axis= 0)
        sigma_final = np.mean( np.array([sigma_freq,sigma_welsh]),axis= 0)
        pops = [0,1,2,3,4,5,6,7]
        np.savetxt(analysis_dict["name"] + name_, np.c_[pops,mean_final, amplitude_final, sigma_final], fmt = '%.2f', header = 'Pops mean_freq amplitude sigma')

def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter_signal(data,fs,lowcut,highcut,order=3):
    filtered_signal = {}
    for i in data:
        filtered_signal[str(i)] = butter_bandpass_filter(data[str(i)][analysis_dict["analysis_start"]:analysis_dict["analysis_end"]]-np.mean(data[str(i)][analysis_dict["analysis_start"]:analysis_dict["analysis_end"]]),lowcut,highcut,fs,order)

    return filtered_signal