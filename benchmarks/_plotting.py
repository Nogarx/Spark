#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def single_neuron_plots(
    spike_times,
    firing_threshold,
    spark_times,
    spark_potentials,
    spark_spikes,
    b2_times,
    b2_potentials,
    b2_spikes,
    title,
) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(8,5), height_ratios=(1,2,1))

    ax[0].eventplot(spike_times, linewidth=3)
    ax[0].eventplot(spike_times, color='k', linestyles='--')
    ax[0].set_ylabel('Input Spikes')
    ax[0].set_ylim(0.25,1.75)
    ax[0].set_yticks([], [])
    ax[0].set_xticks([], [])

    ax[1].plot([0, 100], [firing_threshold, firing_threshold], 'r--', alpha=0.5)
    ax[1].plot(spark_times, spark_potentials, linewidth=3)
    ax[1].plot(b2_times, b2_potentials, 'k--', alpha=0.8)
    ax[1].set_ylabel('Potential [mV]')
    #ax[1].set_ylim(-75,-45)
    ax[1].set_xticks([], [])

    ax[2].eventplot(spark_spikes, linewidths=3)
    ax[2].eventplot(b2_spikes, color='k', linestyles='--', alpha=0.8)
    ax[2].set_ylabel('Output Spikes')
    ax[2].set_yticks([], [])
    ax[2].set_ylim(0.25,1.75)
    ax[2].set_xlabel('Time [ms]')

    for i in range(3):
        ax[i].set_xlim(0, spark_times[-1])

    legend_elements = [
        Line2D([], [], color='k', linestyle='--', label='Brian2'), 
        Line2D([], [], linestyle='solid', linewidth=3, label='Spark'), 
    ]
    fig.legend(handles=legend_elements, ncols=2, bbox_to_anchor=(0.5, 1.0), loc='upper center')

    plt.suptitle(title, y=1.05)
    plt.tight_layout()
    plt.show()

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################