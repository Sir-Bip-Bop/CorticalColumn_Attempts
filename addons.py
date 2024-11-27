import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nest 
from matplotlib.patches import Polygon
from scipy.fft import fft
from scipy.fft import fftfreq


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