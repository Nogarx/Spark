#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
from spark.core import addresses

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class TestMatches:
    """
        A pattern against one address.
    """

    @pytest.mark.parametrize('pattern, address', [
        ('pool.soma', 'pool.soma'),
        ('*_excitatory.soma', 'A_excitatory.soma'),
        ('pool_?.soma', 'pool_a.soma'),
        ('pool_[ab].soma', 'pool_b.soma'),
        ('**.threshold', 'threshold'),
        ('**.threshold', 'pool.soma.threshold'),
        ('pool.**.kernel', 'pool.synapses.kernel'),
        ('pool.*:spikes', 'pool.soma:spikes'),
        ('pool.soma:*', 'pool.soma:spikes'),
        ('__call__:*', '__call__:signal'),
        ('pool.__call__:*', 'pool.__call__:in_spikes'),
    ])
    def test_a_match(self, pattern, address) -> None:
        assert addresses.matches(pattern, address)

    @pytest.mark.parametrize('pattern, address', [
        ('pool.soma', 'pool.soma.threshold'),
        ('*.soma', 'brain.pool.soma'),
        ('pool_?.soma', 'pool_ab.soma'),
        ('**.threshold', 'pool.soma.threshold_tau'),
        ('pool.*:spikes', 'pool.soma.spikes'),
        ('pool.soma.*', 'pool.soma:spikes'),
        ('*:signal', '__call__:signal'),
        ('pool.**:in_spikes', 'pool.__call__:in_spikes'),
    ])
    def test_no_match(self, pattern, address) -> None:
        assert not addresses.matches(pattern, address)

#-----------------------------------------------------------------------------------------------------------------------------------------------#

class TestSelect:
    """
        A pattern against several addresses.
    """

    def test_the_addresses_matched_keep_their_order(self) -> None:
        found = ('b_excitatory.soma', 'a_inhibitory.soma', 'a_excitatory.soma', 'a_excitatory.synapses')
        assert addresses.select('*_excitatory.soma', found) == ('b_excitatory.soma', 'a_excitatory.soma')

    @pytest.mark.parametrize('text, is_pattern', [('pool.soma', False), ('*.soma', True), ('pool_?', True), ('[ab].soma', True)])
    def test_a_pattern_holds_a_wildcard(self, text, is_pattern) -> None:
        assert addresses.is_pattern(text) is is_pattern

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################
