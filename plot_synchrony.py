import matplotlib.pyplot as plt 
import numpy as np 
from matplotlib.widgets import Button, Slider 
from matplotlib.patches import Polygon
import os 
import helpers 
import addons 
import random

def analyse_synchrony(num_neurons,name,bin_width=3,t_r = 2,dt=0.01):
    analysis_interval_start = addons.analysis_dict["analysis_start"]
    analysis_interval_end = addons.analysis_dict["analysis_end"]
    analysis_interval_start_s = addons.analysis_dict["synchrony_start"]
    analysis_interval_end_s = addons.analysis_dict["synchrony_end"]
    
    
    analysis_length = analysis_interval_end - analysis_interval_start
    analysis_length_s = analysis_interval_end_s - analysis_interval_start_s
    pop_activity = {}
    if analysis_length - analysis_length_s < 0:
        print('There is a problem. Synchrony measurement range must be smaller than the other')
        return 

    sd_names, node_ids, data = helpers.__load_spike_times(name,"spike_recorder",analysis_interval_start_s, analysis_interval_end_s)
    
    helper = np.loadtxt(name+"measurements/pop_activities/pop_activity_"+str(0)+".dat")
    sum_array = np.zeros_like(helper)
    for i in range(len(num_neurons)):
        pop_activity[i] = np.loadtxt(name+"measurements/pop_activities/pop_activity_"+str(i)+".dat")
        sum_array = sum_array + pop_activity[i]

    data_s = {}

    for i in data:
        low = np.searchsorted(data[i]["time_ms"],v=analysis_interval_start_s,side="left")
        high = np.searchsorted(data[i]["time_ms"],v=analysis_interval_end_s,side='right')
        data_s[i] = data[i][low:high]


    synchrony_pd = []
    synchrony_chi = []
    irregularity = []
    irregularity_pdf = {}


    super_lvr = []
    lvr_pdf = {}

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
        counts2, bins = np.histogram(times_s[i], bins=int(analysis_length_s/bin_width/2))
    
        synchrony_chi = np.append(synchrony_chi,np.var(counts2)/np.mean(counts2)) 
        synchrony_pd = np.append(synchrony_pd,np.var(counts)/np.mean(counts))

        #Computing irregularity and LvR
        single_irregularity = []
        used_senders = []
        single_lvr = 0
        lvr = []

    mean_pop = sum_array / len(num_neurons)
    sum = 0
    for i, n in enumerate(pop_activity):
        sum = sum + np.var(pop_activity[i])
    chi = np.var(mean_pop) / ((1 / len(num_neurons)) * sum)

    
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
        if np.mean(single_irregularity)==None:
            irregularity = np.append(irregularity,0)
        else:
            irregularity = np.append(irregularity,np.mean(single_irregularity))
        irregularity_pdf[i] = single_irregularity
        if count >= 1000:
            break
    
    #super_lvr = super_lvr[~np.isnan(super_lvr)]
    #irregularity = irregularity[~np.isnan(irregularity)] 
 
    return synchrony_pd, synchrony_chi, irregularity, irregularity_pdf, super_lvr, lvr_pdf, times_s, chi 

list_dirs = os.listdir("data_background_rate")
list_dirs = sorted(list_dirs)
ims = []

synchrony_1= {}
synchrony_2 = {}
irregularity_1 = {}
irregularity_2 = {}
lvr_1 = {}
lvr_2 = {}

bg_rate = np.linspace(4,12,int((12-4)/0.25))

for i, dir in enumerate(list_dirs):
    name = "data_background_rate/" + str(dir)
    neuron_id = np.loadtxt((os.path.join(name,"population_nodeids.dat")),dtype=int)
    num_neurons = []
    for j in range(len(neuron_id)):
        num_neurons = np.append(num_neurons,int(neuron_id[j][1]-neuron_id[j][0]+1))
    num_neurons = num_neurons.astype(int)
    #Correctly reading the number of neurons
    name = "data_background_rate/" + str(dir) + "/"
    synchrony_pd, synchrony_chi, irregularity, irregularity_pdf, lvr, lvr_pdf, times_s, chi = analyse_synchrony(num_neurons,name)
    synchrony_1[bg_rate[i]] = synchrony_pd
    synchrony_2[bg_rate[i]] = synchrony_chi
    irregularity_1[bg_rate[i]] = irregularity
    irregularity_2[bg_rate[i]] = irregularity_pdf
    lvr_1[bg_rate[i]] = lvr 
    lvr_2[bg_rate[i]] = lvr_pdf


bg_rate_ini = 8.13
fs = 18
pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']


############################################################################################################
fig, ax = plt.subplots(figsize=(15,15))
bar = ax.barh(y = pops[::-1], width= synchrony_1[bg_rate[16]][::-1], color = colours[::-1])
#bar = ax.barh(y = pops[::-1], width= synchrony_pd[::-1], color = colours[::-1])
ax.set_xlabel('synchrony', fontsize = fs)
ax.grid(alpha = 0.5)

axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=bg_rate[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    bar = ax.barh(y = pops[::-1], width= synchrony_1[slider.val][::-1], color = colours[::-1])
    ax.set_xlabel('synchrony', fontsize = fs)
    ax.grid(alpha = 0.5)
    fig.canvas.draw_idle()

slider.on_changed(update)

resetax = fig.add_axes([0.8,0.025,0.1,0.04])
button = Button(resetax, 'Reset',hovercolor='0.975')

def reset(event):
    slider.reset()
button.on_clicked(reset)
plt.show()


##############################################################################

fig, ax = plt.subplots(figsize=(15,15))
test = []
colours_box = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
pops_box = pops
label_pos = list(range(len(pops_box),0,-1))

for i in np.arange(len(pops)):
    if len(irregularity_2[bg_rate[16]][i]) == 0: 
        del pops_box[i]
    else:
        test.append(irregularity_2[bg_rate[16]][i])

medianprops = dict(linestyle="-", linewidth=2.5, color="black")
meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

bp = ax.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True,orientation='horizontal')
label_pos = list(range(len(pops_box), 0, -1))
for i in np.arange(len(pops_box)):
    boxX = []
    boxY = []
    box = bp["boxes"][i]
    for j in list(range(5)):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor=colours_box[-i])
    ax.add_patch(boxPolygon)
ax.set_xlabel('irregulatiry', fontsize = fs)
ax.set_yticks(label_pos, pops_box, fontsize=fs)
ax.grid(alpha = 0.5)

help = np.linspace(6,12,int((12-6)/0.25))
axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=help[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    test = []
    pops_box = pops
    label_pos = list(range(len(pops_box),0,-1))

    for i in np.arange(len(pops)):
        if len(irregularity_2[slider.val][i]) == 0: 
            del pops_box[i]
        else:
            test.append(irregularity_2[slider.val][i])

    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

    bp = ax.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True,orientation='horizontal')
    label_pos = list(range(len(pops_box), 0, -1))
    #if len(pops) != len(pops_box):
    #    colours_box = ['#3288bd', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    for i in np.arange(len(pops_box)):
        boxX = []
        boxY = []
        box = bp["boxes"][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colours_box[-i])
        ax.add_patch(boxPolygon)
    ax.set_xlabel('irregulatiry', fontsize = fs)
    ax.set_yticks(label_pos, pops_box, fontsize=fs)
    plt.grid(alpha = 0.5)
    fig.canvas.draw_idle()

slider.on_changed(update)

resetax = fig.add_axes([0.8,0.025,0.1,0.04])
button = Button(resetax, 'Reset',hovercolor='0.975')

def reset(event):
    slider.reset()
button.on_clicked(reset)
plt.show()

##############################################################################
fig, ax = plt.subplots(figsize=(15,15))

pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colours_pdf = list(reversed(colours))
pops_pdf = list(reversed(pops))
irregularity_total = []
for i in irregularity_2[bg_rate[16]]:
    data, bins= np.histogram(irregularity_2[bg_rate[16]][i],density=True,bins=50)
    irregularity_total = np.append(irregularity_total,irregularity_2[bg_rate[16]][i])
    ax.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops_pdf[i], color = colours_pdf[i])

data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
ax.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
ax.legend()
plt.ylabel('P (irregularity)', fontsize = fs)
plt.xlabel('CV ISI', fontsize = fs)
plt.grid(alpha = 0.5)
plt.xlim(0,1.8)
plt.ylim(0,3.6)


axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=bg_rate[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    irregularity_total = []
    for i in irregularity_2[slider.val]:
        data, bins= np.histogram(irregularity_2[slider.val][i],density=True,bins=50)
        irregularity_total = np.append(irregularity_total,irregularity_2[slider.val][i])
        ax.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops_pdf[i], color = colours_pdf[i])

    data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
    ax.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
    ax.legend()
    ax.set_ylabel('P (irregularity)', fontsize = fs)
    ax.set_xlabel('CV ISI', fontsize = fs)
    ax.grid(alpha = 0.5)
    plt.xlim(0,1.8)
    plt.ylim(0,3.6)
    fig.canvas.draw_idle()

slider.on_changed(update)

resetax = fig.add_axes([0.8,0.025,0.1,0.04])
button = Button(resetax, 'Reset',hovercolor='0.975')

def reset(event):
    slider.reset()
button.on_clicked(reset)
plt.show()

##############################################################################

fig, ax = plt.subplots(figsize=(15,15))
test = []
colours_box = ['#3288bd','#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
pops_box = pops
label_pos = list(range(len(pops_box),0,-1))

for i in np.arange(len(pops)):
    if len(lvr_2[bg_rate[16]][i]) == 0: 
        del pops_box[i]
    else:
        test.append(lvr_2[bg_rate[16]][i])

medianprops = dict(linestyle="-", linewidth=2.5, color="black")
meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

bp = ax.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True,orientation='horizontal')
label_pos = list(range(len(pops_box), 0, -1))
for i in np.arange(len(pops_box)):
    boxX = []
    boxY = []
    box = bp["boxes"][i]
    for j in list(range(5)):
        boxX.append(box.get_xdata()[j])
        boxY.append(box.get_ydata()[j])
    boxCoords = list(zip(boxX, boxY))
    boxPolygon = Polygon(boxCoords, facecolor=colours_box[-i])
    ax.add_patch(boxPolygon)
ax.set_xlabel('lvr', fontsize = fs)
ax.set_yticks(label_pos, pops_box, fontsize=fs)
ax.grid(alpha = 0.5)

help = np.linspace(6,12,int((12-6)/0.25))
axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=help[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    test = []
    pops_box = pops
    label_pos = list(range(len(pops_box),0,-1))

    for i in np.arange(len(pops)):
        if len(lvr_2[slider.val][i]) == 0: 
            del pops_box[i]
        else:
            test.append(lvr_2[slider.val][i])

    medianprops = dict(linestyle="-", linewidth=2.5, color="black")
    meanprops = dict(linestyle="--", linewidth=2.5, color="darkgray")

    bp = ax.boxplot(test, 0, "rs",0,medianprops=medianprops,meanprops=meanprops, meanline=True, showmeans=True,orientation='horizontal')
    label_pos = list(range(len(pops_box), 0, -1))
    #if len(pops) != len(pops_box):
    #    colours_box = ['#3288bd', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5']
    for i in np.arange(len(pops_box)):
        boxX = []
        boxY = []
        box = bp["boxes"][i]
        for j in list(range(5)):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        boxCoords = list(zip(boxX, boxY))
        boxPolygon = Polygon(boxCoords, facecolor=colours_box[-i])
        ax.add_patch(boxPolygon)
    ax.set_xlabel('lvr', fontsize = fs)
    ax.set_yticks(label_pos, pops_box, fontsize=fs)
    plt.grid(alpha = 0.5)
    fig.canvas.draw_idle()

slider.on_changed(update)

resetax = fig.add_axes([0.8,0.025,0.1,0.04])
button = Button(resetax, 'Reset',hovercolor='0.975')

def reset(event):
    slider.reset()
button.on_clicked(reset)
plt.show()

##############################################################################
fig, ax = plt.subplots(figsize=(15,15))

pops = ["L23E", "L23I", "L4E", "L4I", "L5E", "L5I", "L6E", "L6I"]
colours = ['#d53e4f', '#f46d43', '#fdae61', '#fee08b', '#e6f598', '#abdda4', '#66c2a5', '#3288bd']
colours_pdf = list(reversed(colours))
pops_pdf = list(reversed(pops))
irregularity_total = []
for i in lvr_2[bg_rate[16]]:
    data, bins= np.histogram(lvr_2[bg_rate[16]][i],density=True,bins=50)
    irregularity_total = np.append(irregularity_total,lvr_2[bg_rate[16]][i])
    ax.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops_pdf[i], color = colours_pdf[i])

data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
ax.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
ax.legend()
plt.ylabel('P (lvr)', fontsize = fs)
plt.xlabel('something', fontsize = fs)
plt.grid(alpha = 0.5)
plt.xlim(0,1.8)
plt.ylim(0,3.6)


axfreq = fig.add_axes([0.25,0.1,0.65,0.03])
axfreq.set_xticks(np.linspace(4,12,int((12-4)/0.25)), minor = False)
slider = Slider(ax = axfreq, label = 'background rate (spikes/s)', valmin=bg_rate[0], valmax=bg_rate[-1], valinit=bg_rate[16],valstep=np.linspace(4,12,int((12-4)/0.25)))

def update(val):
    ax.cla()
    irregularity_total = []
    for i in lvr_2[slider.val]:
        data, bins= np.histogram(lvr_2[slider.val][i],density=True,bins=50)
        irregularity_total = np.append(irregularity_total,lvr_2[slider.val][i])
        ax.plot(bins[:-1],data,alpha=0.3+i*0.1, label = pops_pdf[i], color = colours_pdf[i])

    data_t, bins_s = np.histogram(irregularity_total,density=True,bins=50)
    ax.plot(bins[:-1],data_t, label = 'Total', color = 'black', ls = 'dashed')
   
    ax.legend()
    ax.set_ylabel('P (lvr)', fontsize = fs)
    ax.set_xlabel('soemthing', fontsize = fs)
    ax.grid(alpha = 0.5)
    plt.xlim(0,1.8)
    plt.ylim(0,3.6)
    fig.canvas.draw_idle()

slider.on_changed(update)

resetax = fig.add_axes([0.8,0.025,0.1,0.04])
button = Button(resetax, 'Reset',hovercolor='0.975')

def reset(event):
    slider.reset()
button.on_clicked(reset)
plt.show()