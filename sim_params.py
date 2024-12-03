import os

sim_dict = {
    # The full simulation time is the sum of a presimulation time and the main
    # simulation time.
    # presimulation time (in ms)
    "t_presim": 500.0,
    # simulation time (in ms)
    "t_sim": 5000.0, #1000
    # resolution of the simulation (in ms)
    "sim_resolution": 0.1,
    # list of recording devices, default is 'spike_recorder'. A 'voltmeter' can
    # be added to record membrane voltages of the neurons. Nothing will be
    # recorded if an empty list is given.
    "rec_dev": ["spike_recorder", "voltmeter", "synaptic_ex","synaptic_in"],#["spike_recorder","voltmeter"], 
    # path to save the output data
    "data_path": os.path.join(os.getcwd(), "data_og/"),
    # Seed for NEST
    "rng_seed": 55,
    # Number of threads per MPI process
    #
    # Note that when you scale up the network, the microcircut model
    # may not run correctly if there is < 4 virtual processes
    # (i.e., a thread in an MPI process)
    # If you have 4 or more MPI processes, then you can set this value to 1.
    "local_num_threads": 10,
    # recording interval of the membrane potential (in ms)
    "rec_V_int": 1.0,
    "start" : 500,
    "stop": 5000,
    # if True, data will be overwritten,
    # if False, a NESTError is raised if the files already exist
    "overwrite_files": True,
    # print the time progress. This should only be used when the simulation
    # is run on a local machine.
    "print_time": False,
}