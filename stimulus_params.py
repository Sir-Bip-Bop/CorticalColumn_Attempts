import numpy as np

stim_dict = {
    # optional thalamic input
    # turn thalamic input on or off (True or False)
    "thalamic_input": False,

    "external_input": True,

    "target_pop": [2],

    "input_type": 'pulse_packet', #poisson, gaussian_pulse, gamma, pulse packet
    # start of the thalamic input (in ms)
    "th_start": 500.0,
    # duration of the thalamic input (in ms)
    "th_duration": 5000.0,
    # rate of the thalamic input (in spikes/s)
    "th_rate": 12.0,

    "th_dev": 20,
    # number of thalamic neurons
    "num_th_neurons": 902, #902

    "pulse_number": 50,
    # connection probabilities of the thalamus to the different populations
    # (same order as in 'populations' in 'net_dict')
    "conn_probs_th": np.array([0.0, 0.0, 0.0983, 0.0619, 0.0, 0.0, 0.0512, 0.0196]),
    # mean amplitude of the thalamic postsynaptic potential (in mV),
    # standard deviation will be taken from 'net_dict'
    "PSP_th": 0.15,
    # mean delay of the thalamic input (in ms)
    "delay_th_mean": 1.5,
    # relative standard deviation of the thalamic delay (in ms)
    "delay_th_rel_std": 0.5,
    # optional DC input
    # turn DC input on or off (True or False)
    "dc_input": False,
    # start of the DC input (in ms)
    "dc_start": 650.0,
    # duration of the DC input (in ms)
    "dc_dur": 100.0,
    # amplitude of the DC input (in pA); final amplitude is population-specific
    # and will be obtained by multiplication with 'K_ext'
    "dc_amp": 0.3,

    "spike_weights": 0.7,

    "frequency": 20,

    "activity": 600, #num of spikes in one pulse packet
    "sdev": 1.0, #width of pulse packet (ms)
    "weight": 2.0, #psp amplitude (mV)
    "pulsetime": 1000, #occurrence time (center) of pulse packet(in ms)

}