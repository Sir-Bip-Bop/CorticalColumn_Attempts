import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nest 
import helpers
from matplotlib.patches import Polygon
from scipy.fft import fft
from scipy.fft import fftfreq
import matplotlib.pyplot as plt
import scienceplots
import random
plt.style.use(['science'])


analysis_dict = {
    "analysis_start": 500,
    "analysis_end": 2500,
    "name": "bg_16/", 
    "synchrony_start": 500,
    "synchrony_end": 3500,
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
        irregularity_pdf[i] = irregularity
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
