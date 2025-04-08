import os
import warnings

import helpers
import nest
import numpy as np


class Network:
    """Provides functions to setup NEST, to create and connect all nodes of
    the network, to simulate, and to evaluate the resulting spike data.

    Instantiating a Network object derives dependent parameters and already
    initializes the NEST kernel.

    Parameters
    ---------
    sim_dict
        Dictionary containing all parameters specific to the simulation
        (see: ``sim_params.py``).
    net_dict
         Dictionary containing all parameters specific to the neuron and
         network models (see: ``network_params.py``).
    stim_dict
        Optional dictionary containing all parameter specific to the stimulus
        (see: ``stimulus_params.py``)

    """

    def __init__(self, sim_dict, net_dict, stim_dict=None,path=None,bg_rate =None,ref_period=None,matrix=None,PSP_mean=None,th_rate=0,stim_time=None):
        self.sim_dict = sim_dict
        self.net_dict = net_dict
        self.stim_dict = stim_dict
        self.PSP_mean = PSP_mean
        # data directory
        if path == None:
            self.data_path = sim_dict["data_path"]
        else:
            self.data_path = path
        
        self.th_rate = th_rate
        self.matrix_delays = matrix
        self.ref_period = ref_period
        self.bg_rate = bg_rate
        self.stim_time = stim_time
        if nest.Rank() == 0: #this function returns the MPI rank of the local process, which is 0 in the case where the network has not yet been initialised
            if os.path.isdir(self.data_path):
                message = "  Directory already existed."
                if self.sim_dict["overwrite_files"]:
                    message += " Old data will be overwritten."
            else:
                os.mkdir(self.data_path)
                message = "  Directory has been created."
            print("Data will be written to: {}\n{}\n".format(self.data_path, message))

        # derive parameters based on input dictionaries
        self.__derive_parameters(PSP_mean=PSP_mean)

        # initialize the NEST kernel
        self.__setup_nest()

    def create(self):
        """Creates all network nodes.

        Neuronal populations and recording and stimulation devices are created.

        """
        self.__create_neuronal_populations()
        if len(self.sim_dict["rec_dev"]) > 0:
            self.__create_recording_devices()
        if self.net_dict["poisson_input"]:
            self.__create_poisson_bg_input()
        if self.stim_dict["thalamic_input"]:
            self.__create_thalamic_stim_input()
        if self.stim_dict["external_input"]:
            if self.stim_time != None:
                self.__create_external_stim_input()
        if self.stim_dict["dc_input"]:
            self.__create_dc_stim_input()

    def connect(self):
        """Connects the network.

        Recurrent connections among neurons of the neuronal populations are
        established, and recording and stimulation devices are connected.

        The ``self.__connect_*()`` functions use ``nest.Connect()`` calls which
        set up the postsynaptic connectivity.
        Since the introduction of the 5g kernel in NEST 2.16.0 the full
        connection infrastructure including presynaptic connectivity is set up
        afterwards in the preparation phase of the simulation.
        The preparation phase is usually induced by the first
        ``nest.Simulate()`` call.
        For including this phase in measurements of the connection time,
        we induce it here explicitly by calling ``nest.Prepare()``.

        """
        self.__connect_neuronal_populations()

        if len(self.sim_dict["rec_dev"]) > 0:
            self.__connect_recording_devices()
        if self.net_dict["poisson_input"]:
            self.__connect_poisson_bg_input()
        if self.stim_dict["thalamic_input"]:
            self.__connect_thalamic_stim_input()
        if self.stim_dict["external_input"]:
            if self.stim_time != None:
                self.__connect_external_stim_input()
        if self.stim_dict["dc_input"]:
            self.__connect_dc_stim_input()

        nest.Prepare()
        nest.Cleanup()

    def simulate(self, t_sim):
        """Simulates the microcircuit.

        Parameters
        ----------
        t_sim
            Simulation time (in ms).

        """
        if nest.Rank() == 0:
            print("Simulating {} ms.".format(t_sim))

        nest.Simulate(t_sim)

    def evaluate(self, raster_plot_interval, firing_rates_interval,binned=False,M= [20,20,20,20,20,20,20,20],std= [1,1,1,1,1,1,1,1],trial=10):
        """Displays simulation results.

        Creates a spike raster plot.
        Calculates the firing rate of each population and displays them as a
        box plot.

        Parameters
        ----------
        raster_plot_interval
            Times (in ms) to start and stop loading spike times for raster plot
            (included).
        firing_rates_interval
            Times (in ms) to start and stop lading spike times for computing
            firing rates (included).

        Returns
        -------
            None

        """
        if nest.Rank() == 0:
            print("Interval to plot spikes: {} ms".format(raster_plot_interval))
            pop_activity = helpers.plot_raster(
                self.data_path,
                "spike_recorder",
                raster_plot_interval[0],
                raster_plot_interval[1],
                self.net_dict["N_scaling"],
                binned,
                M,
                std,
                trial
            )

            print("Interval to compute firing rates: {} ms".format(firing_rates_interval))
            helpers.firing_rates(self.data_path, "spike_recorder", firing_rates_interval[0], firing_rates_interval[1])
            helpers.boxplot(self.data_path, self.net_dict["populations"],trial)
        return pop_activity
    def __derive_parameters(self,PSP_mean=None):
        """
        Derives and adjusts parameters and stores them as class attributes.
        """
        self.num_pops = len(self.net_dict["populations"])

        # total number of synapses between neuronal populations before scaling
        full_num_synapses = helpers.num_synapses_from_conn_probs(
            self.net_dict["conn_probs"], self.net_dict["full_num_neurons"], self.net_dict["full_num_neurons"]
        )

        # scaled numbers of neurons and synapses
        self.num_neurons = np.round((self.net_dict["full_num_neurons"] * self.net_dict["N_scaling"])).astype(int)
        self.num_synapses = np.round(
            (full_num_synapses * self.net_dict["N_scaling"] * self.net_dict["K_scaling"])
        ).astype(int)
        self.ext_indegrees = np.round((self.net_dict["K_ext"] * self.net_dict["K_scaling"])).astype(int)

        # conversion from PSPs to PSCs - blind faith that this is the correct way to do it
        PSC_over_PSP = helpers.postsynaptic_potential_to_current(
            self.net_dict["neuron_params"]["C_m"],
            self.net_dict["neuron_params"]["tau_m"],
            self.net_dict["neuron_params"]["tau_syn"],
        )
        if PSP_mean != None:
            PSC_matrix_mean = PSP_mean * PSC_over_PSP
        else:
            PSC_matrix_mean = self.net_dict["PSP_matrix_mean"] * PSC_over_PSP
        PSC_ext = self.net_dict["PSP_exc_mean"] * PSC_over_PSP

        # DC input compensates for potentially missing Poisson input
        if self.net_dict["poisson_input"]:
            DC_amp = np.zeros(self.num_pops)
        else:
            if nest.Rank() == 0:
                warnings.warn("DC input created to compensate missing Poisson input.\n")
            DC_amp = helpers.dc_input_compensating_poisson(
                self.net_dict["bg_rate"], self.net_dict["K_ext"], self.net_dict["neuron_params"]["tau_syn"], PSC_ext
            )

        # adjust weights and DC amplitude if the indegree is scaled
        if self.net_dict["K_scaling"] != 1:
            PSC_matrix_mean, PSC_ext, DC_amp = helpers.adjust_weights_and_input_to_synapse_scaling(
                self.net_dict["full_num_neurons"],
                full_num_synapses,
                self.net_dict["K_scaling"],
                PSC_matrix_mean,
                PSC_ext,
                self.net_dict["neuron_params"]["tau_syn"],
                self.net_dict["full_mean_rates"],
                DC_amp,
                self.net_dict["poisson_input"],
                self.net_dict["bg_rate"],
                self.net_dict["K_ext"],
            )

        # store final parameters as class attributes
        self.weight_matrix_mean = PSC_matrix_mean
        self.weight_ext = PSC_ext
        self.DC_amp = DC_amp

        # thalamic input
        if self.stim_dict["thalamic_input"]:
            num_th_synapses = helpers.num_synapses_from_conn_probs(
                self.stim_dict["conn_probs_th"], self.stim_dict["num_th_neurons"], self.net_dict["full_num_neurons"]
            )[0]
            self.weight_th = self.stim_dict["PSP_th"] * PSC_over_PSP
            if self.net_dict["K_scaling"] != 1:
                num_th_synapses *= self.net_dict["K_scaling"]
                self.weight_th /= np.sqrt(self.net_dict["K_scaling"])
            self.num_th_synapses = np.round(num_th_synapses).astype(int)

        if nest.Rank() == 0:
            message = ""
            if self.net_dict["N_scaling"] != 1:
                message += "Neuron numbers are scaled by a factor of {:.3f}.\n".format(self.net_dict["N_scaling"])
            if self.net_dict["K_scaling"] != 1:
                message += "Indegrees are scaled by a factor of {:.3f}.".format(self.net_dict["K_scaling"])
                message += "\n  Weights and DC input are adjusted to compensate.\n"
            print(message)

    def __setup_nest(self):
        """Initializes the NEST kernel.

        Reset the NEST kernel and pass parameters to it.
        """
        nest.ResetKernel()

        nest.local_num_threads = self.sim_dict["local_num_threads"]
        nest.resolution = self.sim_dict["sim_resolution"]
        nest.rng_seed = self.sim_dict["rng_seed"]
        nest.overwrite_files = self.sim_dict["overwrite_files"]
        nest.print_time = self.sim_dict["print_time"]

        rng_seed = nest.rng_seed
        vps = nest.total_num_virtual_procs

        if nest.Rank() == 0:
            print("RNG seed: {}".format(rng_seed))
            print("Total number of virtual processes: {}".format(vps))

    def __create_neuronal_populations(self):
        """Creates the neuronal populations.

        The neuronal populations are created and the parameters are assigned
        to them. The initial membrane potential of the neurons is drawn from
        normal distributions dependent on the parameter ``V0_type``.

        The first and last neuron id of each population is written to file.
        """
        if nest.Rank() == 0:
            print("Creating neuronal populations.")

        self.pops = []
        if self.ref_period != None:
            t_ref = self.ref_period
        else:
            t_ref = self.net_dict["neuron_params"]["t_ref"]
        for i in np.arange(self.num_pops):
            population = nest.Create(self.net_dict["neuron_model"], self.num_neurons[i])

            population.set(
                tau_syn_ex=self.net_dict["neuron_params"]["tau_syn"],
                tau_syn_in=self.net_dict["neuron_params"]["tau_syn"],
                E_L=self.net_dict["neuron_params"]["E_L"],
                V_th=self.net_dict["neuron_params"]["V_th"],
                V_reset=self.net_dict["neuron_params"]["V_reset"],
                t_ref=nest.random.normal(t_ref,self.net_dict["neuron_params"]["t_std"],),
                #t_ref = t_ref,
                I_e=self.DC_amp[i],
            )

            if self.net_dict["V0_type"] == "optimized":
                population.set(
                    V_m=nest.random.normal(
                        self.net_dict["neuron_params"]["V0_mean"]["optimized"][i],
                        self.net_dict["neuron_params"]["V0_std"]["optimized"][i],
                    )
                )
            elif self.net_dict["V0_type"] == "original":
                population.set(
                    V_m=nest.random.normal(
                        self.net_dict["neuron_params"]["V0_mean"]["original"],
                        self.net_dict["neuron_params"]["V0_std"]["original"],
                    )
                )
            else:
                raise ValueError("V0_type is incorrect. " + 'Valid options are "optimized" and "original".')

            self.pops.append(population)

        # write node ids to file
        if nest.Rank() == 0:
            fn = os.path.join(self.data_path, "population_nodeids.dat")
            with open(fn, "w+") as f:
                for pop in self.pops:
                    f.write("{} {}\n".format(pop[0].global_id, pop[-1].global_id))

    def __create_recording_devices(self):
        """Creates one recording device of each kind per population.

        Only devices which are given in ``sim_dict['rec_dev']`` are created.

        """
        if nest.Rank() == 0:
            print("Creating recording devices.")

        if "spike_recorder" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("  Creating spike recorders.")
            sd_dict = {"record_to": "ascii", "label": os.path.join(self.data_path, "spike_recorder")}
            self.spike_recorders = nest.Create("spike_recorder", n=self.num_pops, params=sd_dict)

        if "voltmeter" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("  Creating voltmeters.")
            vm_dict = {
                "interval": self.sim_dict["rec_V_int"],
                "record_to": "ascii",
                "record_from": ["V_m"],
                "label": os.path.join(self.data_path, "voltmeter"),
                "start": self.sim_dict["start"],
                "stop": self.sim_dict["stop"],
            }
            self.voltmeters = nest.Create("voltmeter", n=self.num_pops, params=vm_dict)
        if "synaptic_ex" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("Creating ammeters.")
            ex_dict = {
                "interval": self.sim_dict["rec_V_int"],
                "record_to": "ascii",
                "record_from": ["I_syn_ex"],
                "label": os.path.join(self.data_path, "ex_current"),
                "start": self.sim_dict["start"],
                "stop": self.sim_dict["stop"],
            }
            self.ex_ammeters = nest.Create("voltmeter",n=self.num_pops,params=ex_dict)

        if "synaptic_in" in self.sim_dict["rec_dev"]:
            if nest.Rank() == 0:
                print("Creating ammeters.")
            ex_dict = {
                "interval": self.sim_dict["rec_V_int"],
                "record_to": "ascii",
                "record_from": ["I_syn_in"],
                "label": os.path.join(self.data_path, "in_current"),
                "start": self.sim_dict["start"],
                "stop": self.sim_dict["stop"],
            }
            self.in_ammeters = nest.Create("voltmeter",n=self.num_pops,params=ex_dict)
        

    def __create_poisson_bg_input(self):
        """Creates the Poisson generators for ongoing background input if
        specified in ``network_params.py``.

        If ``poisson_input`` is ``False``, DC input is applied for compensation
        in ``create_neuronal_populations()``.

        """
        if nest.Rank() == 0:
            print("Creating Poisson generators for background input.")

        self.ppgs = {}
        self.poisson_bg_input = nest.Create("poisson_generator", n=self.num_pops)
        if self.bg_rate:
            self.poisson_bg_input.rate = self.bg_rate * self.ext_indegrees
        else:
            self.poisson_bg_input.rate = self.net_dict["bg_rate"] * self.ext_indegrees
    def __create_external_stim_input(self):
        """Creates the external neuronal population if specified in
        ``stim_dict``.

        """
        if nest.Rank() == 0:
            print("Creating external input for external stimulation.")

        ppg_args = {"pulse_times": [self.stim_time], "activity": self.stim_dict["activity"], "sdev": self.stim_dict["sdev"]}
        n_neurons = self.num_neurons[self.stim_dict["target_pop"]]
        for i in range(len(n_neurons)):
            self.ppgs[i] = nest.Create('pulsepacket_generator',n_neurons[i],ppg_args)
        
    def __create_thalamic_stim_input(self):
        """Creates the thalamic neuronal population if specified in
        ``stim_dict``.

        Each neuron of the thalamic population is supposed to transmit the same
        Poisson spike train to all of its targets in the cortical neuronal population,
        and spike trains elicited by different thalamic neurons should be statistically
        independent.
        In NEST, this is achieved with a single Poisson generator connected to all
        thalamic neurons which are of type ``parrot_neuron``;
        Poisson generators send independent spike trains to each of their targets and
        parrot neurons just repeat incoming spikes.

        Note that the number of thalamic neurons is not scaled with
        ``N_scaling``.

        """
        if nest.Rank() == 0:
            print("Creating thalamic input for external stimulation.")

        self.thalamic_population = nest.Create("parrot_neuron", n=self.stim_dict["num_th_neurons"])

        if self.th_rate != 0:
            rate = self.th_rate
        else:
            rate = self.stim_dict["th_rate"]

        if self.stim_dict["input_type"] == 'poisson':
            self.poisson_th = nest.Create("poisson_generator")
            self.poisson_th.set(
                rate=rate,
                start=self.stim_dict["th_start"],
                stop=(self.stim_dict["th_start"] + self.stim_dict["th_duration"]),
            )
        if self.stim_dict["input_type"] == "gaussian_pulse":
            self.poisson_th = nest.Create("pulsepacket_generator")
            self.poisson_th.set(
                start = self.stim_dict["th_start"],
                stop = (self.stim_dict["th_start"] + self.stim_dict["th_duration"]),
                pulse_times = np.linspace(self.stim_dict["th_start"],self.stim_dict["th_start"]+self.stim_dict["th_duration"],num=self.stim_dict["pulse_number"]),
                #pulse_times = [500,510,520],
                activity = int(self.stim_dict["th_rate"]),
                sdev = self.stim_dict["th_dev"],
            )
        if self.stim_dict["input_type"] == "gamma":
            self.poisson_th = nest.Create("sinusoidal_gamma_generator")
            self.poisson_th.set(
                start = self.stim_dict["th_start"],
                stop = self.stim_dict["th_start"] + self.stim_dict["th_duration"],
                rate = rate,
                frequency = self.stim_dict["frequency"],
            )

    def __create_dc_stim_input(self):
        """Creates DC generators for external stimulation if specified
        in ``stim_dict``.

        The final amplitude is the ``stim_dict['dc_amp'] * net_dict['K_ext']``.

        """
        dc_amp_stim = self.stim_dict["dc_amp"] * self.net_dict["K_ext"]

        if nest.Rank() == 0:
            print("Creating DC generators for external stimulation.")

        dc_dict = {
            "amplitude": dc_amp_stim,
            "start": self.stim_dict["dc_start"],
            "stop": self.stim_dict["dc_start"] + self.stim_dict["dc_dur"],
        }
        self.dc_stim_input = nest.Create("dc_generator", n=self.num_pops, params=dc_dict)

    def __connect_neuronal_populations(self):
        """Creates the recurrent connections between neuronal populations."""
        if nest.Rank() == 0:
            print("Connecting neuronal populations recurrently.")

        if self.matrix_delays !=None:
            delay_matrix_mean = self.matrix_delays
        else:
            delay_matrix_mean = self.net_dict["delay_matrix_mean"]

        for i, target_pop in enumerate(self.pops):
            for j, source_pop in enumerate(self.pops):
                if self.num_synapses[i][j] >= 0.0:
                    conn_dict_rec = {"rule": "fixed_total_number", "N": self.num_synapses[i][j]}

                    if self.weight_matrix_mean[i][j] < 0:
                        w_min = -np.inf
                        w_max = 0.0
                    else:
                        w_min = 0.0
                        w_max = np.inf

                    syn_dict = {
                        "synapse_model": "static_synapse",
                        "weight": nest.math.redraw(
                            nest.random.normal(
                                mean=self.weight_matrix_mean[i][j],
                                std=abs(self.weight_matrix_mean[i][j] * self.net_dict["weight_rel_std"]),
                            ),
                            min=w_min,
                            max=w_max,
                        ),
                        "delay": nest.math.redraw(
                            nest.random.normal(
                                mean=delay_matrix_mean[i][j],
                                std=(delay_matrix_mean[i][j] * self.net_dict["delay_rel_std"]),
                            ),
                            # resulting minimum delay is equal to resolution, see:
                            # https://nest-simulator.readthedocs.io/en/latest/nest_behavior
                            # /random_numbers.html#rounding-effects-when-randomizing-delays
                            min=nest.resolution - 0.5 * nest.resolution,
                            max=np.inf,
                        ),
                    }

                    nest.Connect(source_pop, target_pop, conn_spec=conn_dict_rec, syn_spec=syn_dict)

    def __connect_recording_devices(self):
        """Connects the recording devices to the microcircuit."""
        if nest.Rank == 0:
            print("Connecting recording devices.")
        
        conn_spec = {'rule': 'fixed_total_number','N':1000}

        for i, target_pop in enumerate(self.pops):
            print(target_pop)
            if "spike_recorder" in self.sim_dict["rec_dev"]:
                nest.Connect(target_pop, self.spike_recorders[i])
            if "voltmeter" in self.sim_dict["rec_dev"]:
                nest.Connect(self.voltmeters[i], target_pop)
            if "synaptic_ex" in self.sim_dict["rec_dev"]:
                nest.Connect(self.ex_ammeters[i],target_pop)
            if "synaptic_in" in self.sim_dict["rec_dev"]:
                nest.Connect(self.in_ammeters[i],target_pop,)

    def __connect_poisson_bg_input(self):
        """Connects the Poisson generators to the microcircuit."""
        if nest.Rank() == 0:
            print("Connecting Poisson generators for background input.")

        for i, target_pop in enumerate(self.pops):
            conn_dict_poisson = {"rule": "all_to_all"}

            syn_dict_poisson = {
                "synapse_model": "static_synapse",
                "weight": self.weight_ext,
                "delay": self.net_dict["delay_poisson"],
            }

            nest.Connect(self.poisson_bg_input[i], target_pop, conn_spec=conn_dict_poisson, syn_spec=syn_dict_poisson)

    def __connect_thalamic_stim_input(self):
        """Connects the thalamic input to the neuronal populations."""
        if nest.Rank() == 0:
            print("Connecting thalamic input.")

        # connect Poisson input to thalamic population
        nest.Connect(self.poisson_th, self.thalamic_population)

        # connect thalamic population to neuronal populations
        for i, target_pop in enumerate(self.pops):
            conn_dict_th = {"rule": "fixed_total_number", "N": self.num_th_synapses[i]}

            syn_dict_th = {
                "weight": nest.math.redraw(
                    nest.random.normal(mean=self.weight_th, std=self.weight_th * self.net_dict["weight_rel_std"]),
                    min=0.0,
                    max=np.inf,
                ),
                "delay": nest.math.redraw(
                    nest.random.normal(
                        mean=self.stim_dict["delay_th_mean"],
                        std=(self.stim_dict["delay_th_mean"] * self.stim_dict["delay_th_rel_std"]),
                    ),
                    # resulting minimum delay is equal to resolution, see:
                    # https://nest-simulator.readthedocs.io/en/latest/nest_behavior
                    # /random_numbers.html#rounding-effects-when-randomizing-delays
                    min=nest.resolution - 0.5 * nest.resolution,
                    max=np.inf,
                ),
            }

            nest.Connect(self.thalamic_population, target_pop, conn_spec=conn_dict_th, syn_spec=syn_dict_th)
    def __connect_external_stim_input(self):
        """Connects the external pulse packet to the neuronal populations."""

        if nest.Rank() == 0:
            print("Connecting DC generators.")
        j = 0
        for i, target_pop in enumerate(self.pops):
            if i in self.stim_dict["target_pop"]:
                nest.Connect(self.ppgs[j], target_pop, "one_to_one", syn_spec= {"weight": self.stim_dict["weight"]})
                j = j+1

    def __connect_dc_stim_input(self):
        """Connects the DC generators to the neuronal populations."""

        if nest.Rank() == 0:
            print("Connecting DC generators.")

        for i, target_pop in enumerate(self.pops):
            nest.Connect(self.dc_stim_input[i], target_pop)