#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import numpy as np

# NOTE: These methods were automatically generated but were validated against a verified implementation with some 
# manual examples and a couple of randomly generated cases.

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def _next_isi(spikes, idx, t_start, t_end, prev_isi):
	"""
	Compute the ISI following the spike at index `idx`.

	Boundary handling:
	- Before first spike (idx < 0)
	- Between spikes
	- After last spike

	Parameters
	----------
	spikes : ndarray
		Spike times.
	idx : int
		Index of the current spike.
	prev_isi : float
		Previously used ISI (needed for right boundary handling).
	"""
	n = len(spikes)

	# Before first spike: estimate ISI using first two spikes or start boundary
	if idx < 0:
		if n > 1:
			return max(spikes[0] - t_start, spikes[1] - spikes[0])
		return spikes[0] - t_start

	# Between spikes: standard forward ISI
	if idx < n - 1:
		return spikes[idx + 1] - spikes[idx]

	# After last spike: compare distance to end with previous ISI
	if n > 1:
		return max(t_end - spikes[-1], prev_isi)
	return t_end - spikes[-1]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def isi_val(a, b, MRTS):
	"""
	Instantaneous ISI distance.

	|a - b| normalized by the larger ISI (or MRTS).
	"""
	return abs(a - b) / max(MRTS, max(a, b))

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def isi_distance(s1, s2, t_start, t_end, MRTS=0.0):
    """
    Compute the ISI (Inter-Spike-Interval) distance between two spike trains.

    The ISI distance measures the instantaneous mismatch between the
    local inter-spike intervals (ISIs) of two spike trains and averages
    this mismatch over time.

    Parameters
    ----------
    s1, s2 : array_like
        Sorted spike times for spike train 1 and 2.
    t_start, t_end : float
        Start and end times of the observation window.
    MRTS : float, optional
        Minimum Relevant Time Scale. Acts as a lower bound in the
        normalization to avoid singularities for very small ISIs.

    Returns
    -------
    float
        Time-averaged ISI distance over [t_start, t_end].

    References
    ----------
    Kreuz et al., Scholarpedia: Measures of spike train synchrony.
    """

    # Ensure NumPy float arrays
    s1 = np.asarray(s1, float)
    s2 = np.asarray(s2, float)

    # Spike indices (start before first spike)
    i1 = i2 = -1

    # Initial ISIs at t_start
    nu1 = _next_isi(s1, i1, t_start, t_end, 0.0)
    nu2 = _next_isi(s2, i2, t_start, t_end, 0.0)
    
 	# Last processed time
    last_t = t_start       
    # Accumulated integral of ISI mismatch
    value = 0.0             

    # Initial instantaneous ISI distance
    curr = isi_val(nu1, nu2, MRTS)

    # Merge and sort all spike events
    events = np.sort(np.concatenate((s1, s2)))

    # Iterate over spike times
    for t in events:
        # Integrate ISI distance over the interval [last_t, t]
        value += curr * (t - last_t)

        # Update ISI of spike train 1 if spike occurs
        if i1 + 1 < len(s1) and t == s1[i1 + 1]:
            i1 += 1
            nu1 = _next_isi(s1, i1, t_start, t_end, nu1)

        # Update ISI of spike train 2 if spike occurs
        if i2 + 1 < len(s2) and t == s2[i2 + 1]:
            i2 += 1
            nu2 = _next_isi(s2, i2, t_start, t_end, nu2)

        # Recompute instantaneous ISI distance
        curr = isi_val(nu1, nu2, MRTS)
        last_t = t

    # Final integration until t_end
    value += curr * (t_end - last_t)

    # Normalize by total duration
    return value / (t_end - t_start)

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

def mid_dist(spike_time, spike_train, N, start_index, t_start, t_end):
	"""
	Compute the minimal absolute temporal distance between a given spike time
	and a reference spike train, constrained to a local forward search.

	This function exploits the fact that `spike_train` is sorted in time.
	The search starts at `start_index` and stops as soon as distances increase,
	ensuring O(local) complexity rather than a full scan.

	Parameters
	----------
	spike_time : float
		Time of the spike for which the distance is evaluated.
	spike_train : array-like
		Sorted spike times of the reference train.
	N : int
		Number of spikes in `spike_train`.
	start_index : int
		Index at which the local search starts (usually last known closest spike).
	t_start, t_end : float
		Boundary times used for edge correction.

	Returns
	-------
	float
		Minimal distance to the closest spike or boundary.
	"""

	# Initialize with distance to left boundary (edge correction)
	d = abs(spike_time - t_start)

	# Clamp start index to valid range
	if start_index < 0:
		start_index = 0

	# Local forward search for nearest spike
	current_idx = start_index
	while current_idx < N:
		d_temp = abs(spike_time - spike_train[current_idx])

		# Since spike_train is sorted, distances increase after the minimum
		if d_temp > d:
			return d
		else:
			d = d_temp

		current_idx += 1

	# Compare with right boundary (edge correction)
	d_temp = abs(t_end - spike_time)
	return d if d_temp > d else d_temp

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def dist_t(isi1, isi2, s1, s2, MRTS, RI):
	"""
	Compute the instantaneous SPIKE-distance value at a given time.

	This corresponds to Eq. (19–21) in the SPIKE-distance formulation and
	combines local spike-time differences weighted by inter-spike intervals.

	Parameters
	----------
	isi1, isi2 : float
		Local inter-spike intervals of spike trains 1 and 2.
	s1, s2 : float
		Instantaneous distances to nearest spikes in the opposite train.
	MRTS : float
		Minimum Relevant Time Scale (regularization parameter).
	RI : bool
		If True, use rate-independent formulation.

	Returns
	-------
	float
		Instantaneous SPIKE-distance value.
	"""

	# Mean ISI and lower bound regularization
	meanISI = 0.5 * (isi1 + isi2)
	limitedISI = max(MRTS, meanISI)

	if RI:
		# Rate-independent SPIKE distance
		return 0.5 * (s1 + s2) / limitedISI
	else:
		# Classical SPIKE distance
		return 0.5 * (s1 * isi2 + s2 * isi1) / (meanISI * limitedISI)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

def spike_distance(t1, t2, t_start, t_end, MRTS=0., RI=False):
	"""
	Compute the SPIKE distance between two spike trains over a time interval.

	This is a faithful pure-Python translation of the original Cython
	event-driven algorithm. The distance is obtained by integrating the
	instantaneous SPIKE-distance over time.

	Parameters
	----------
	t1, t2 : array-like
		Sorted spike times of the two spike trains.
	t_start, t_end : float
		Start and end times of the observation interval.
	MRTS : float, optional
		Minimum Relevant Time Scale.
	RI : bool, optional
		Enable rate-independent SPIKE distance.

	Returns
	-------
	float
		Time-averaged SPIKE distance over [t_start, t_end].
	"""

	N1 = len(t1)
	N2 = len(t2)

	assert N1 > 0
	assert N2 > 0

	# Accumulated integral value
	spike_value = 0.0

	# Last integration boundary
	t_last = t_start

	# -------------------------------------------------
	# 1. Auxiliary spikes (edge correction)
	# -------------------------------------------------
	if N1 > 1:
		aux1_0 = min(t_start, 2 * t1[0] - t1[1])
		aux1_1 = max(t_end,   2 * t1[-1] - t1[-2])
	else:
		aux1_0, aux1_1 = t_start, t_end

	if N2 > 1:
		aux2_0 = min(t_start, 2 * t2[0] - t2[1])
		aux2_1 = max(t_end,   2 * t2[-1] - t2[-2])
	else:
		aux2_0, aux2_1 = t_start, t_end

	# -------------------------------------------------
	# 2. Initialization of state variables
	# -------------------------------------------------
	t_p1 = t_start if t1[0] == t_start else aux1_0
	t_p2 = t_start if t2[0] == t_start else aux2_0

	# ---- Train 1 initialization ----
	if t1[0] > t_start:
		t_f1 = t1[0]
		dt_f1 = mid_dist(t_f1, t2, N2, 0, aux2_0, aux2_1)
		isi1 = max(t_f1 - t_start, t1[1] - t1[0]) if N1 > 1 else t_f1 - t_start
		dt_p1 = dt_f1
		s1 = dt_p1
		index1 = -1
	else:
		t_f1 = t1[1] if N1 > 1 else t_end
		dt_f1 = mid_dist(t_f1, t2, N2, 0, aux2_0, aux2_1)
		dt_p1 = mid_dist(t_p1, t2, N2, 0, aux2_0, aux2_1)
		isi1 = t_f1 - t1[0]
		s1 = dt_p1
		index1 = 0

	# ---- Train 2 initialization ----
	if t2[0] > t_start:
		t_f2 = t2[0]
		dt_f2 = mid_dist(t_f2, t1, N1, 0, aux1_0, aux1_1)
		isi2 = max(t_f2 - t_start, t2[1] - t2[0]) if N2 > 1 else t_f2 - t_start
		dt_p2 = dt_f2
		s2 = dt_p2
		index2 = -1
	else:
		t_f2 = t2[1] if N2 > 1 else t_end
		dt_f2 = mid_dist(t_f2, t1, N1, 0, aux1_0, aux1_1)
		dt_p2 = mid_dist(t_p2, t1, N1, 0, aux1_0, aux1_1)
		isi2 = t_f2 - t2[0]
		s2 = dt_p2
		index2 = 0

	# Initial instantaneous distance
	y_start = dist_t(isi1, isi2, s1, s2, MRTS, RI)

	# -------------------------------------------------
	# 3. Main event-driven integration loop
	# -------------------------------------------------
	while index1 + index2 < N1 + N2 - 2:

		# Determine which train has the next spike
		train1_next = (index1 < N1 - 1) and (t_f1 < t_f2 or index2 == N2 - 1)
		train2_next = (index2 < N2 - 1) and (t_f1 > t_f2 or index1 == N1 - 1)

		if train1_next:
			# Advance train 1
			index1 += 1
			s1 = dt_f1 * (t_f1 - t_p1) / isi1
			dt_p1 = dt_f1
			t_p1 = t_f1

			t_f1 = t1[index1 + 1] if index1 < N1 - 1 else aux1_1
			t_curr = t_p1

			# Linear interpolation of s2
			s2 = (dt_p2 * (t_f2 - t_p1) + dt_f2 * (t_p1 - t_p2)) / isi2

			y_end = dist_t(isi1, isi2, s1, s2, MRTS, RI)
			spike_value += 0.5 * (y_start + y_end) * (t_curr - t_last)

			if index1 < N1 - 1:
				dt_f1 = mid_dist(t_f1, t2, N2, index2, aux2_0, aux2_1)
				isi1 = t_f1 - t_p1
				s1 = dt_p1
			else:
				dt_f1 = dt_p1
				isi1 = max(t_end - t1[-1], t1[-1] - t1[-2]) if N1 > 1 else t_end - t1[-1]
				s1 = dt_p1

			y_start = dist_t(isi1, isi2, s1, s2, MRTS, RI)

		elif train2_next:
			# Symmetric update for train 2
			index2 += 1
			s2 = dt_f2 * (t_f2 - t_p2) / isi2
			dt_p2 = dt_f2
			t_p2 = t_f2

			t_f2 = t2[index2 + 1] if index2 < N2 - 1 else aux2_1
			t_curr = t_p2

			s1 = (dt_p1 * (t_f1 - t_p2) + dt_f1 * (t_p2 - t_p1)) / isi1

			y_end = dist_t(isi1, isi2, s1, s2, MRTS, RI)
			spike_value += 0.5 * (y_start + y_end) * (t_curr - t_last)

			if index2 < N2 - 1:
				dt_f2 = mid_dist(t_f2, t1, N1, index1, aux1_0, aux1_1)
				isi2 = t_f2 - t_p2
				s2 = dt_p2
			else:
				dt_f2 = dt_p2
				isi2 = max(t_end - t2[-1], t2[-1] - t2[-2]) if N2 > 1 else t_end - t2[-1]
				s2 = dt_p2

			y_start = dist_t(isi1, isi2, s1, s2, MRTS, RI)

		else:
			# Simultaneous spikes (perfect synchrony)
			index1 += 1
			index2 += 1
			t_p1 = t_f1
			t_p2 = t_f2
			dt_p1 = dt_p2 = 0.0
			t_curr = t_f1

			spike_value += 0.5 * y_start * (t_curr - t_last)
			y_start = 0.0

			t_f1 = t1[index1 + 1] if index1 < N1 - 1 else aux1_1
			t_f2 = t2[index2 + 1] if index2 < N2 - 1 else aux2_1

		t_last = t_curr

	# -------------------------------------------------
	# 4. Final integration segment
	# -------------------------------------------------
	y_end = dist_t(isi1, isi2, dt_f1, dt_f2, MRTS, RI)
	spike_value += 0.5 * (y_start + y_end) * (t_end - t_last)

	return spike_value / (t_end - t_start)


#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################