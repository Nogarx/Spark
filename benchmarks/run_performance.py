#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import numpy as np

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

from _brian2 import (
    brian_adex_performance_model_1_non_interactive, brian_adex_performance_model_1_interactive_numpy_mock,
    brian_adex_performance_model_2_non_interactive, brian_adex_performance_model_2_interactive_numpy_mock,
    brian_adex_performance_model_3_non_interactive, brian_adex_performance_model_3_interactive_numpy_mock
)
import brian2 as b2
from _spark import (
    spark_adex_performance_model_1, spark_adex_performance_model_2, spark_adex_performance_model_3
)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def run_brian_mock_benchmark(name, model_fn, k_times, t_max, sim_seconds, sim_reps):
    b2_times_inter_mock = []
    print(f'Brian2 (interactive mock): Running {sim_seconds}s simulations, {sim_reps} times each')
    for k in k_times:
        times = model_fn(
            sim_repetitions = sim_reps,
            t_max = t_max,
            k_time = k * b2.ms,
            units = UNITS,
            ker_density = KER_DENSITY,
            synapse_strength = SYNAPSE_STRENGTH,
            dt = DT * b2.ms,
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
        b2_times_inter_mock.append(times)
        print(f'{k} ms steps:\t{np.mean(times):.4f} ± {np.std(times, ddof=1):.4f}')
    np.save(f'./npy_files/b2_times_inter_mock_t{sim_seconds}r{sim_reps}_{name}.npy', b2_times_inter_mock)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_brian_cpp_benchmark(name, model_fn, k_times, t_max, sim_seconds, sim_reps):
    b2_times_non_inter = []
    print(f'Brian2 (non-interactive): Running {sim_seconds}s simulations, {sim_reps} times each')
    for k in k_times:
        times = model_fn(
            sim_repetitions = sim_reps,
            t_max = t_max,
            k_time = k * b2.ms,
            units = UNITS,
            ker_density = KER_DENSITY,
            synapse_strength = SYNAPSE_STRENGTH,
            dt = DT * b2.ms,
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
        b2_times_non_inter.append(times)
        print(f'{k} ms steps:\t{np.mean(times):.4f} ± {np.std(times, ddof=1):.4f}')
    np.save(f'./npy_files/b2_times_non_inter_t{sim_seconds}r{sim_reps}_{name}.npy', b2_times_non_inter)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def run_spark_benchmark(name, model_fn, k_steps, t_max, sim_seconds, sim_reps):
    spark_times = []
    print(f'Spark: Running {sim_seconds}s simulations, {sim_reps} times each')
    for k in k_steps:
        # Get clean state
        times = model_fn(
            sim_repetitions = sim_reps,
            t_steps = t_max, 
            k_steps = k,
            units = UNITS,
            ker_density = KER_DENSITY,
            dt = DT,
            synapse_strength = SYNAPSE_STRENGTH,
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
        )
        spark_times.append(times)
        print(f'{k} steps:\t{np.mean(times):.4f} ± {np.std(times, ddof=1):.4f}')
    np.save(f'./npy_files/spark_times_t{sim_seconds}r{sim_reps}_{name}.npy', spark_times)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

BENCHMARK_NAME = 'v1'
SIM_SECONDS = 5.0
SIM_REPS = 25
k_times = [
    #0.1, # [0.1ms] Too slow for interactive
    1.0, # [1ms]
    2.0, # [2ms]
    5.0, # [5ms]
    10.0, # [10ms]
    50.0, # [50ms]
]
t_max_brian = SIM_SECONDS * b2.second
k_steps = [
    #1, # [0.1ms] Too slow for interactive
	10, # [1ms]
	20, # [2ms]
	50, # [5ms]
	100, # [10ms]
	500, # [50ms]
]
t_max_spark = int((SIM_SECONDS * 1000) / DT)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('\n')
print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 1')
print('-' * 25 + '\n')

run_brian_mock_benchmark(BENCHMARK_NAME, brian_adex_performance_model_1_interactive_numpy_mock, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_1_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_1, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 2')
print('-' * 25 + '\n')

run_brian_mock_benchmark(BENCHMARK_NAME, brian_adex_performance_model_2_interactive_numpy_mock, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_2_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_2, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 3')
print('-' * 25 + '\n')

run_brian_mock_benchmark(BENCHMARK_NAME, brian_adex_performance_model_3_interactive_numpy_mock, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_3_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_3, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################


BENCHMARK_NAME = 'v2'
SIM_SECONDS = 500
SIM_REPS = 100
k_times = [
    0.1, # [0.1ms]
    1.0, # [1ms]
    2.0, # [2ms]
    5.0, # [5ms]
    10.0, # [10ms]
    50.0, # [50ms]
]
t_max_brian = SIM_SECONDS * b2.second
k_steps = [
    1, # [0.1ms]
    10, # [1ms]
    20, # [2ms]
    50, # [5ms]
    100, # [10ms]
    500, # [50ms]
]
t_max_spark = int((SIM_SECONDS * 1000) / DT)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 1')
print('-' * 25 + '\n')

run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_1_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_1, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 2')
print('-' * 25 + '\n')

run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_2_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_2, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

print('-' * 25)
print(f'Benchmark {BENCHMARK_NAME}: Model 3')
print('-' * 25 + '\n')

run_brian_cpp_benchmark(BENCHMARK_NAME, brian_adex_performance_model_3_non_interactive, k_times, t_max_brian, SIM_SECONDS, SIM_REPS)
run_spark_benchmark(BENCHMARK_NAME, spark_adex_performance_model_3, k_steps, t_max_spark, SIM_SECONDS, SIM_REPS)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################