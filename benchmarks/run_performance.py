#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import numpy as np
import brian2 as b2
import os
from _brian2 import (
    brian2_adex_performance_model_1_cpp, brian2_adex_performance_model_1_cpp_step, brian2_adex_performance_model_1_numpy,
    brian2_adex_performance_model_2_cpp, brian2_adex_performance_model_2_cpp_step, brian2_adex_performance_model_2_numpy,
    brian2_adex_performance_model_3_cpp, brian2_adex_performance_model_3_cpp_step, brian2_adex_performance_model_3_numpy
)

from _spark import (
    spark_adex_performance_model_1, spark_adex_performance_model_2, spark_adex_performance_model_3
)
from tqdm import tqdm

# NOTE: Brian2 numpy simulations throws a bunch of warnings about orphaned objects.
# It seems that its not any leaking memory (we do an extra manually gc collection step just in case)
# but the errors are really annoying
b2.utils.logger.get_logger('brian2.core.base').log_level_error()


MAIN_FOLDER = f'./performance/'
os.makedirs(MAIN_FOLDER, exist_ok=True)
SIM_REPETITIONS = 10
MODEL_REPETITIONS = 25
MAX_TIME = 10000.0

#-----------------------------------------------------------------------------------------------------------------------------------------------#

UNITS = 1024
KER_DENSITY = 0.15
DT = 0.1
MEMBRANE_TIME_SCALE_TAU_M = 5.0
MEMBRANE_RESISTANCE_R = 500.0
V_REST = -70.0
V_RESET = -51.0
RHEOBASE_THRESHOLD_V_RH = -50.0
SHARPNESS_DELTA_T = 2.0
ADAPTATION_VOLTAGE_COUPLING_A = 0.5
ADAPTATION_TIME_CONSTANT_TAU_W = 10.0
SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B = 5.0
FIRING_THRESHOLD_V_SPIKE = -30.0
SYNAPSE_STRENGTH = 30.0 / UNITS
FREQ_SPIKE = 30 / 1000

SPARK_PARAMS = {
	'potential_rest': V_REST, 
	'potential_reset': V_RESET, 
	'rheobase_threshold': RHEOBASE_THRESHOLD_V_RH,
	'adaptation_subthreshold': ADAPTATION_VOLTAGE_COUPLING_A / 1000,	# nS -> µS 
	'adaptation_delta': SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B / 1000, # pA -> nA 
	'threshold': FIRING_THRESHOLD_V_SPIKE,
	'spike_slope': SHARPNESS_DELTA_T, 
	'adaptation_tau': ADAPTATION_TIME_CONSTANT_TAU_W, 
	'potential_tau': MEMBRANE_TIME_SCALE_TAU_M , 
	'resistance': MEMBRANE_RESISTANCE_R,
}

B2_PARAMS = {
	'v_rest': V_REST * b2.mV, 
	'v_reset': V_RESET * b2.mV, 
	'v_rheobase': RHEOBASE_THRESHOLD_V_RH * b2.mV,
	'a': ADAPTATION_VOLTAGE_COUPLING_A * b2.nS, 
	'b': SPIKE_TRIGGERED_ADAPTATION_INCREMENT_B * b2.pA, 
	'firing_threshold': FIRING_THRESHOLD_V_SPIKE * b2.mV,
	'delta_T': SHARPNESS_DELTA_T * b2.mV, 
	'tau_w': ADAPTATION_TIME_CONSTANT_TAU_W * b2.ms, 
	'tau_m': MEMBRANE_TIME_SCALE_TAU_M * b2.ms, 
	'R': MEMBRANE_RESISTANCE_R * b2.Mohm,
}

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def run_brian_numpy(model_fn, sim_step_times, t_max):
    benchmark_data = {}
    for sim_step in tqdm(sim_step_times):
        b2_compile_times, b2_times = [], []
        for _ in range(MODEL_REPETITIONS):
            compile_time, times = model_fn(
                sim_repetitions = SIM_REPETITIONS,
                units = UNITS,
                ker_density = KER_DENSITY,
                synapse_strength = SYNAPSE_STRENGTH,
                freq_spike = FREQ_SPIKE,
                t_max = t_max * b2.ms,
                sim_step = sim_step * b2.ms,
                dt = DT * b2.ms,
                model_params = B2_PARAMS,
                kernels = None,
            )
            b2_times.append(times)
            b2_compile_times.append(compile_time)
        benchmark_data[sim_step] = {
            'sim': np.array(b2_times),
            'compile': np.array(b2_compile_times),
        }
    return benchmark_data

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_cpp_step(model_fn, sim_step_times, t_max):
    benchmark_data = {}
    for sim_step in tqdm(sim_step_times):
        b2_compile_times, b2_times = [], []
        for _ in range(MODEL_REPETITIONS):
            compile_time, times = model_fn(
                sim_repetitions = SIM_REPETITIONS,
                units = UNITS,
                ker_density = KER_DENSITY,
                synapse_strength = SYNAPSE_STRENGTH,
                freq_spike = FREQ_SPIKE,
                t_max = t_max  * b2.ms,
                sim_step = sim_step * b2.ms,
                dt = DT * b2.ms,
                model_params = B2_PARAMS,
                kernels = None,
                delete = True,
                directory='_adex_cpp',
            )
            b2_times.append(times)
            b2_compile_times.append(compile_time)
        benchmark_data[sim_step] = {
            'sim': np.array(b2_times),
            'compile': np.array(b2_compile_times),
        }
    return benchmark_data

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_cpp(model_fn, t_max):
    b2_compile_times, b2_times = [], []
    for _ in tqdm(range(MODEL_REPETITIONS)):
        compile_time, times = model_fn(
            sim_repetitions = SIM_REPETITIONS,
            units = UNITS,
            ker_density = KER_DENSITY,
            synapse_strength = SYNAPSE_STRENGTH,
            freq_spike = FREQ_SPIKE,
            t_max = t_max * b2.ms,
            dt = DT * b2.ms,
            model_params = B2_PARAMS,
            kernels = None,
            delete = True,
            directory=f'_adex_cpp',
        )
        b2_times.append(times)
        b2_compile_times.append(compile_time)
    return {
        'sim': np.array(b2_times),
        'compile': np.array(b2_compile_times),
    }

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_spark(model_fn, k_steps, t_max):
    benchmark_data = {}
    for k in tqdm(k_steps):
        b2_compile_times, b2_times = [], []
        for _ in range(MODEL_REPETITIONS):
            compile_time, times = model_fn(
                sim_repetitions = SIM_REPETITIONS,
                t_max = t_max, 
                sim_steps = k,
                units = UNITS,
                ker_density = KER_DENSITY,
                synapse_strength = SYNAPSE_STRENGTH,
                freq_spike = FREQ_SPIKE,
                dt = DT,
                model_params = SPARK_PARAMS,
            )
            b2_times.append(times)
            b2_compile_times.append(compile_time)
        benchmark_data[k] = {
            'sim': np.array(b2_times),
            'compile': np.array(b2_compile_times),
        }
    return benchmark_data

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

# Compile steps
k_times = [
    #0.1, # [0.1ms] Too slow for interactive
    1.0, # [1ms]
    2.0, # [2ms]
    5.0, # [5ms]
    10.0, # [10ms]
    50.0, # [50ms]
]
k_steps = [
    #1, # [0.1ms] Too slow for interactive
	10, # [1ms]
	20, # [2ms]
	50, # [5ms]
	100, # [10ms]
	500, # [50ms]
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

MODEL_NAME = 'adex_1'
print('\n')
print('-' * 25)
print(f'Benchmark: {MODEL_NAME}')
print('-' * 25 + '\n')

print('Running Brian2 (numpy)')
b2_numpy_data = run_brian_numpy(brian2_adex_performance_model_1_numpy, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_numpy.npy'), b2_numpy_data)

print('Running Brian2 (c++ step)')
b2_cpp_step_data = run_brian_cpp_step(brian2_adex_performance_model_1_cpp_step, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp_step.npy'), b2_cpp_step_data)

print('Running Brian2 (c++)')
b2_cpp_data = run_brian_cpp(brian2_adex_performance_model_1_cpp, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp.npy'), b2_cpp_data)

print('Running Spark')
spark_data = run_spark(spark_adex_performance_model_1, k_steps, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark.npy'), spark_data)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

MODEL_NAME = 'adex_2'
print('-' * 25)
print(f'Benchmark: {MODEL_NAME}')
print('-' * 25 + '\n')

print('Running Brian2 (numpy)')
b2_numpy_data = run_brian_numpy(brian2_adex_performance_model_2_numpy, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_numpy.npy'), b2_numpy_data)

print('Running Brian2 (c++ step)')
b2_cpp_step_data = run_brian_cpp_step(brian2_adex_performance_model_2_cpp_step, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp_step.npy'), b2_cpp_step_data)

print('Running Brian2 (c++)')
b2_cpp_data = run_brian_cpp(brian2_adex_performance_model_2_cpp, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp.npy'), b2_cpp_data)


print('Running Spark')
spark_data = run_spark(spark_adex_performance_model_2, k_steps, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark.npy'), spark_data)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

MODEL_NAME = 'adex_3'
print('-' * 25)
print(f'Benchmark: {MODEL_NAME}')
print('-' * 25 + '\n')

print('Running Brian2 (numpy)')
b2_numpy_data = run_brian_numpy(brian2_adex_performance_model_3_numpy, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_numpy.npy'), b2_numpy_data)

print('Running Brian2 (c++ step)')
b2_cpp_step_data = run_brian_cpp_step(brian2_adex_performance_model_3_cpp_step, k_times, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp_step.npy'), b2_cpp_step_data)

print('Running Brian2 (c++)')
b2_cpp_data = run_brian_cpp(brian2_adex_performance_model_3_cpp, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_cpp.npy'), b2_cpp_data)

print('Running Spark')
spark_data = run_spark(spark_adex_performance_model_3, k_steps, MAX_TIME)
np.save(os.path.join(MAIN_FOLDER, f'{MODEL_NAME}_spark.npy'), spark_data)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################