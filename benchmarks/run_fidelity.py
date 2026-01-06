#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import numpy as np
from _spark import simulate_LIF_model_spark, simulate_AdEx_model_spark, simulate_HH_model_spark
from _brian2 import simulate_LIF_model_brian, simulate_AdEx_model_brian, simulate_HH_model_brian
from _distances import isi_distance, spike_distance
import brian2 as b2
from tqdm import tqdm
import os

MAIN_FOLDER = f'./fidelity/'
os.makedirs(MAIN_FOLDER, exist_ok=True)
REPETITIONS = 100
MAX_TIME = 1000.0

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def shuffle_spike_train(train, t_max, dt):
	shuffle_train = np.zeros((int(t_max/dt),))
	shuffle_train[(train / dt).astype(int)] = 1
	np.random.shuffle(shuffle_train)
	shuffle_train = np.argwhere(shuffle_train == 1).reshape(-1) * dt
	return shuffle_train

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def rmse(x, y):
	return np.sqrt(np.mean((x-y)**2))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def ou_current(time, dt, tau, sigma, mu=0.0):
    steps = int(time / dt)
    alpha = np.exp(-dt / tau)
    noise_scale = sigma * np.sqrt(1 - alpha**2)

    I = np.empty(steps)
    I[0] = mu
    for t in range(1, steps):
        I[t] = alpha * I[t-1] + noise_scale * np.random.randn() + (1-alpha) * mu
    return I

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Common neuron model parameters
np.random.seed(42)
DT = 0.1
V_REST = -70.0
V_RESET = -65.0
FIRING_THRESHOLD = -50
MEMBRANE_RESISTANCE = 10.0
MEMBRANE_TIME_SCALE = 8.0
ABSOLUTE_REFRACTORY_PERIOD = 2.0
MODEL_NAME = 'LIF'
current_fn = lambda time: ou_current(time, DT, 10, 2, mu=1)
INPUT_CURRENTS = current_fn(100.0)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_pre_current.npy'), INPUT_CURRENTS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Run brian for visual comparison
b2_times, b2_potentials, b2_spikes = simulate_LIF_model_brian(
	currents = INPUT_CURRENTS,
	t_max = len(INPUT_CURRENTS)*DT,
	dt = DT,
	v_rest = V_REST * b2.mV,
	v_reset = V_RESET * b2.mV,
	firing_threshold = FIRING_THRESHOLD * b2.mV,
	membrane_resistance = MEMBRANE_RESISTANCE * b2.Mohm,
	membrane_time_scale = MEMBRANE_TIME_SCALE * b2.ms,
	abs_refractory_period = ABSOLUTE_REFRACTORY_PERIOD * b2.ms
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_times.npy'), b2_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_potentials.npy'), b2_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_spikes.npy'), b2_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Run spark for visual comparison
spark_times, spark_potentials, spark_spikes = simulate_LIF_model_spark(
    currents = INPUT_CURRENTS,
	dt = DT,
	potential_rest = V_REST,
	potential_reset = V_RESET,
	potential_tau = MEMBRANE_TIME_SCALE,
	resistance = MEMBRANE_RESISTANCE,
	threshold = FIRING_THRESHOLD,
	cooldown = ABSOLUTE_REFRACTORY_PERIOD,
    offset = V_REST,
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_times.npy'), spark_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_potentials.npy'), spark_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_spikes.npy'), spark_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Run both models several times to measure the difference.
metrics = {
	'isi_dist': [],
	'spike_dist': [],
	'pot_rmse': [],
	'rand_isi_dist': [],
	'rand_spike_dist': [],
}
for i in tqdm(range(REPETITIONS)):
	INPUT_CURRENTS = current_fn(MAX_TIME)
	b2_times, b2_potentials, b2_spikes = simulate_LIF_model_brian(
		currents = INPUT_CURRENTS,
		t_max = len(INPUT_CURRENTS)*DT,
		dt = DT,
		v_rest = V_REST * b2.mV,
		v_reset = V_RESET * b2.mV,
		firing_threshold = FIRING_THRESHOLD * b2.mV,
		membrane_resistance = MEMBRANE_RESISTANCE * b2.Mohm,
		membrane_time_scale = MEMBRANE_TIME_SCALE * b2.ms,
		abs_refractory_period = ABSOLUTE_REFRACTORY_PERIOD * b2.ms,
		delete = i==(REPETITIONS-1)
	)
	spark_times, spark_potentials, spark_spikes = simulate_LIF_model_spark(
		currents = INPUT_CURRENTS,
		dt = DT,
		potential_rest = V_REST,
		potential_reset = V_RESET,
		potential_tau = MEMBRANE_TIME_SCALE,
		resistance = MEMBRANE_RESISTANCE,
		threshold = FIRING_THRESHOLD,
		cooldown = ABSOLUTE_REFRACTORY_PERIOD,
		offset = V_REST,
	)
	metrics['isi_dist'].append( isi_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['spike_dist'].append( spike_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['pot_rmse'].append( rmse(b2_potentials, spark_potentials) )
	# Random shuffle for comparison
	shuffle_b2_spikes = shuffle_spike_train(b2_spikes, MAX_TIME, DT)
	shuffle_spark_spikes = shuffle_spike_train(spark_spikes, MAX_TIME, DT)
	metrics['rand_isi_dist'].append( isi_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )
	metrics['rand_spike_dist'].append( spike_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )

for k, v in metrics.items():
	metrics[k] = np.array(v).reshape(-1)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_metrics.npy'), metrics)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Common neuron model parameters
np.random.seed(42+1)
DT = 0.1
MEMBRANE_TIME_SCALE_TAU_M = 5.0
MEMBRANE_RESISTANCE_R = 500.0
V_REST = -70.0
V_RESET = -51.0
RHEOBASE_THRESHOLD_V_RH = -50.0
SHARPNESS_DELTA_T = 2.0
ADAPTATION_VOLTAGE_COUPLING_A = 0.5
ADAPTATION_TIME_CONSTANT_TAU_W = 100.0
SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B = 7.0
FIRING_THRESHOLD_V_SPIKE = -30.0
MODEL_NAME = 'AdEx'
current_fn = lambda time: 0.02*ou_current(time, DT, 10, 2, mu=1)
INPUT_CURRENTS = current_fn(100.0)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_pre_current.npy'), INPUT_CURRENTS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

b2_times, b2_potentials, b2_spikes = simulate_AdEx_model_brian(
	currents = INPUT_CURRENTS,
	t_max = len(INPUT_CURRENTS)*DT,
	dt = DT,
	tau_m = MEMBRANE_TIME_SCALE_TAU_M * b2.ms,
	R = MEMBRANE_RESISTANCE_R * b2.Mohm,
	v_rest = V_REST * b2.mV,
	v_reset = V_RESET * b2.mV,
	v_rheobase = RHEOBASE_THRESHOLD_V_RH * b2.mV,
	a = ADAPTATION_VOLTAGE_COUPLING_A * b2.nS,
	b = SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B * b2.pA,
	firing_threshold = FIRING_THRESHOLD_V_SPIKE * b2.mV,
	delta_T = SHARPNESS_DELTA_T * b2.mV,
	tau_w = ADAPTATION_TIME_CONSTANT_TAU_W * b2.ms,
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_times.npy'), b2_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_potentials.npy'), b2_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_spikes.npy'), b2_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

spark_times, spark_potentials, spark_spikes = simulate_AdEx_model_spark(
    currents = INPUT_CURRENTS,
	dt = DT,
	potential_rest = V_REST,
	potential_reset = V_RESET,
	potential_tau = MEMBRANE_TIME_SCALE_TAU_M,
	resistance = MEMBRANE_RESISTANCE_R,
	threshold = FIRING_THRESHOLD_V_SPIKE,
	rheobase_threshold = RHEOBASE_THRESHOLD_V_RH,
	spike_slope = SHARPNESS_DELTA_T,
	adaptation_tau = ADAPTATION_TIME_CONSTANT_TAU_W,
	adaptation_delta = SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B / 1000, # pA -> nA
	adaptation_subthreshold = ADAPTATION_VOLTAGE_COUPLING_A / 1000,	# nS -> µS
    offset = V_REST,
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_times.npy'), spark_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_potentials.npy'), spark_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_spikes.npy'), spark_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Run both models several times to measure the difference.
metrics = {
	'isi_dist': [],
	'spike_dist': [],
	'pot_rmse': [],
	'rand_isi_dist': [],
	'rand_spike_dist': [],
}
for i in tqdm(range(REPETITIONS)):
	INPUT_CURRENTS = current_fn(MAX_TIME)
	b2_times, b2_potentials, b2_spikes = simulate_AdEx_model_brian(
		currents = INPUT_CURRENTS,
		t_max = len(INPUT_CURRENTS)*DT,
		dt = DT,
		tau_m = MEMBRANE_TIME_SCALE_TAU_M * b2.ms,
		R = MEMBRANE_RESISTANCE_R * b2.Mohm,
		v_rest = V_REST * b2.mV,
		v_reset = V_RESET * b2.mV,
		v_rheobase = RHEOBASE_THRESHOLD_V_RH * b2.mV,
		a = ADAPTATION_VOLTAGE_COUPLING_A * b2.nS,
		b = SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B * b2.pA,
		firing_threshold = FIRING_THRESHOLD_V_SPIKE * b2.mV,
		delta_T = SHARPNESS_DELTA_T * b2.mV,
		tau_w = ADAPTATION_TIME_CONSTANT_TAU_W * b2.ms,
		delete = i==(REPETITIONS-1)
	)
	spark_times, spark_potentials, spark_spikes = simulate_AdEx_model_spark(
		currents = INPUT_CURRENTS,
		dt = DT,
		potential_rest = V_REST,
		potential_reset = V_RESET,
		potential_tau = MEMBRANE_TIME_SCALE_TAU_M,
		resistance = MEMBRANE_RESISTANCE_R,
		threshold = FIRING_THRESHOLD_V_SPIKE,
		rheobase_threshold = RHEOBASE_THRESHOLD_V_RH,
		spike_slope = SHARPNESS_DELTA_T,
		adaptation_tau = ADAPTATION_TIME_CONSTANT_TAU_W,
		adaptation_delta = SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B / 1000, # pA -> nA
		adaptation_subthreshold = ADAPTATION_VOLTAGE_COUPLING_A / 1000,	# nS -> µS
		offset = V_REST,
	)
	metrics['isi_dist'].append( isi_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['spike_dist'].append( spike_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['pot_rmse'].append( rmse(b2_potentials, spark_potentials) )
	# Random shuffle for comparison
	shuffle_b2_spikes = shuffle_spike_train(b2_spikes, MAX_TIME, DT)
	shuffle_spark_spikes = shuffle_spike_train(spark_spikes, MAX_TIME, DT)
	metrics['rand_isi_dist'].append( isi_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )
	metrics['rand_spike_dist'].append( spike_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )

for k, v in metrics.items():
	metrics[k] = np.array(v).reshape(-1)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_metrics.npy'), metrics)


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Common neuron model parameters
np.random.seed(42+2)
DT = 0.05
CM = 1
E_LEAK = 10.6
E_NA = 115
E_K = -12
G_LEAK = 0.3
G_NA = 120
G_K = 36
THRESHOLD = 30
MODEL_NAME = 'HH'
current_fn = lambda time: 1.5*ou_current(time, DT, 10, 2, mu=1)
INPUT_CURRENTS = current_fn(100.0)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_pre_current.npy'), INPUT_CURRENTS)

b2_times, b2_potentials, b2_spikes = simulate_HH_model_brian(
	currents = INPUT_CURRENTS,
	t_max = len(INPUT_CURRENTS)*DT,
	dt = DT,
	c_m = CM * b2.ufarad, #/(b2.cm**2)
	e_leak = E_LEAK * b2.mV,
	e_k = E_K * b2.mV,
	e_na = E_NA * b2.mV,
	g_leak = G_LEAK * b2.msiemens,
	g_na = G_NA  * b2.msiemens,
	g_k = G_K  * b2.msiemens,
	threshold = THRESHOLD * b2.mV,
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_times.npy'), b2_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_potentials.npy'), b2_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_b2_spikes.npy'), b2_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

spark_times, spark_potentials, spark_spikes = simulate_HH_model_spark(
    currents = INPUT_CURRENTS,
    dt = DT,
	c_m = CM,
	e_leak = E_LEAK,
	e_k = E_K,
	e_na = E_NA,
	g_leak = G_LEAK,
	g_na = G_NA,
	g_k = G_K,
    threshold = THRESHOLD,
    offset = -70,
)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_times.npy'), spark_times)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_potentials.npy'), spark_potentials)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark_spikes.npy'), spark_spikes)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

# Run both models several times to measure the difference.
metrics = {
	'isi_dist': [],
	'spike_dist': [],
	'pot_rmse': [],
	'rand_isi_dist': [],
	'rand_spike_dist': [],
}
for i in tqdm(range(REPETITIONS)):
	INPUT_CURRENTS = current_fn(MAX_TIME)
	b2_times, b2_potentials, b2_spikes = simulate_HH_model_brian(
		currents = INPUT_CURRENTS,
		t_max = len(INPUT_CURRENTS)*DT,
		dt = DT,
		c_m = CM * b2.ufarad, #/(b2.cm**2)
		e_leak = E_LEAK * b2.mV,
		e_k = E_K * b2.mV,
		e_na = E_NA * b2.mV,
		g_leak = G_LEAK * b2.msiemens,
		g_na = G_NA  * b2.msiemens,
		g_k = G_K  * b2.msiemens,
		threshold = THRESHOLD * b2.mV,
		delete= i==(REPETITIONS-1)
	)
	spark_times, spark_potentials, spark_spikes = simulate_HH_model_spark(
		currents = INPUT_CURRENTS,
		dt = DT,
		c_m = CM,
		e_leak = E_LEAK,
		e_k = E_K,
		e_na = E_NA,
		g_leak = G_LEAK,
		g_na = G_NA,
		g_k = G_K,
		threshold = THRESHOLD,
		offset = -70,
	)
	metrics['isi_dist'].append( isi_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['spike_dist'].append( spike_distance(b2_spikes, spark_spikes, 0, MAX_TIME) )
	metrics['pot_rmse'].append( rmse(b2_potentials, spark_potentials) )
	# Random shuffle for comparison
	shuffle_b2_spikes = shuffle_spike_train(b2_spikes, MAX_TIME, DT)
	shuffle_spark_spikes = shuffle_spike_train(spark_spikes, MAX_TIME, DT)
	metrics['rand_isi_dist'].append( isi_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )
	metrics['rand_spike_dist'].append( spike_distance(shuffle_b2_spikes, shuffle_spark_spikes, 0, MAX_TIME) )

for k, v in metrics.items():
	metrics[k] = np.array(v).reshape(-1)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_metrics.npy'), metrics)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################