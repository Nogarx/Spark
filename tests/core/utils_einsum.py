
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
import numpy as np
import typing as tp
import spark
import dataclasses as dc
np.random.seed(42)
from spark.core.utils import get_einsum_dot_string, get_einsum_dot_red_string, get_einsum_dot_exp_string

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test_success = [
    (
        'Test_r_1',
        get_einsum_dot_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'right',
        'cd,abcd->ab',
	),
    (
        'Test_r_2',
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'right',
        'abcd,cd->ab',
	),
    (
        'Test_r_3',
        get_einsum_dot_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'right',
        'cd,abcd->ab',
	),
    (
        'Test_r_4',
        get_einsum_dot_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'right',
        'abcd,cd->ab',
	),
    (
        'Test_r_5',
        get_einsum_dot_string,
    	(5,), 
        (2, 3, 4, 5), 
        'right',
        'd,abcd->abc',
	),
    (
        'Test_r_6',
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(5,), 
        'right',
        'abcd,d->abc',
	),
    (
        'Test_l_1',
        get_einsum_dot_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'left',
        'ab,abcd->cd',
	),
    (
        'Test_l_2',
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'left',
        'abcd,ab->cd',
	),
    (
        'Test_l_3',
        get_einsum_dot_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'left',
        'ab,abcd->cd',
	),
    (
        'Test_l_4',
        get_einsum_dot_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'left',
        'abcd,ab->cd',
	),
    (
        'Test_l_5',
        get_einsum_dot_string,
    	(2,), 
        (2, 3, 4, 5), 
        'left',
        'a,abcd->bcd',
	),
    (
        'Test_l_6',
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(2,), 
        'left',
        'abcd,a->bcd',
	),	
          (
        'Test_r_1',
        get_einsum_dot_red_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'right',
        'cd,abcd->cd',
	),
    (
        'Test_r_2',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'right',
        'abcd,cd->cd',
	),
    (
        'Test_r_3',
        get_einsum_dot_red_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'right',
        'cd,abcd->cd',
	),
    (
        'Test_r_4',
        get_einsum_dot_red_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'right',
        'abcd,cd->cd',
	),
    (
        'Test_r_5',
        get_einsum_dot_red_string,
    	(5,), 
        (2, 3, 4, 5), 
        'right',
        'd,abcd->d',
	),
    (
        'Test_r_6',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(5,), 
        'right',
        'abcd,d->d',
	),
    (
        'Test_l_1',
        get_einsum_dot_red_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'left',
        'ab,abcd->ab',
	),
    (
        'Test_l_2',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'left',
        'abcd,ab->ab',
	),
    (
        'Test_l_3',
        get_einsum_dot_red_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'left',
        'ab,abcd->ab',
	),
    (
        'Test_l_4',
        get_einsum_dot_red_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'left',
        'abcd,ab->ab',
	),
    (
        'Test_l_5',
        get_einsum_dot_red_string,
    	(2,), 
        (2, 3, 4, 5), 
        'left',
        'a,abcd->a',
	),
    (
        'Test_l_6',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(2,), 
        'left',
        'abcd,a->a',
	),	
    (
        'Test_r_1',
        get_einsum_dot_exp_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'right',
        'cd,abcd->abcd',
	),
    (
        'Test_r_2',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'right',
        'abcd,cd->abcd',
	),
    (
        'Test_r_3',
        get_einsum_dot_exp_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'right',
        'cd,abcd->abcd',
	),
    (
        'Test_r_4',
        get_einsum_dot_exp_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'right',
        'abcd,cd->abcd',
	),
    (
        'Test_r_5',
        get_einsum_dot_exp_string,
    	(5,), 
        (2, 3, 4, 5), 
        'right',
        'd,abcd->abcd',
	),
    (
        'Test_r_6',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(5,), 
        'right',
        'abcd,d->abcd',
	),
    (
        'Test_l_1',
        get_einsum_dot_exp_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'left',
        'ab,abcd->abcd',
	),
    (
        'Test_l_2',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'left',
        'abcd,ab->abcd',
	),
    (
        'Test_l_3',
        get_einsum_dot_exp_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'left',
        'ab,abcd->abcd',
	),
    (
        'Test_l_4',
        get_einsum_dot_exp_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'left',
        'abcd,ab->abcd',
	),
    (
        'Test_l_5',
        get_einsum_dot_exp_string,
    	(2,), 
        (2, 3, 4, 5), 
        'left',
        'a,abcd->abcd',
	),
    (
        'Test_l_6',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2,), 
        'left',
        'abcd,a->abcd',
	),	
    (
        'Test_n_1',
        get_einsum_dot_exp_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'none',
        'ab,cdef->abcdef',
	),
    (
        'Test_n_2',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'none',
        'abcd,ef->abcdef',
	),
    (
        'Test_n_3',
        get_einsum_dot_exp_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'none',
        'ab,cdef->abcdef',
	),
    (
        'Test_n_4',
        get_einsum_dot_exp_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'none',
        'abcd,ef->abcdef',
	),
    (
        'Test_n_5',
        get_einsum_dot_exp_string,
    	(2,), 
        (2, 3, 4, 5), 
        'none',
        'a,bcde->abcde',
	),
    (
        'Test_n_6',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2,), 
        'none',
        'abcd,e->abcde',
	),]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, method, x, y, side, expected', data_test_success)
def test_spark_config_success(name, method, x, y, side, expected) -> None:
    """
        Einsum methods success validation.
    """
    result = method(x, y, side=side)
    assert result == expected
    
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

data_test_failure = [
    (
        'Test_l_1', 
        get_einsum_dot_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_2', 
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'left',
	),
    (
        'Test_l_3', 
        get_einsum_dot_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'left',
	),
    (
        'Test_l_4', 
        get_einsum_dot_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'left',
	),
    (
        'Test_l_5', 
        get_einsum_dot_string,
    	(5,), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_6', 
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(5,), 
        'left',
	),
    (
        'Test_r_1', 
        get_einsum_dot_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_2', 
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'right',
	),
    (
        'Test_r_3', 
        get_einsum_dot_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'right',
	),
    (
        'Test_r_4', 
        get_einsum_dot_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'right',
	),
    (
        'Test_r_5', 
        get_einsum_dot_string,
    	(2,), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_6', 
        get_einsum_dot_string,
        (2, 3, 4, 5), 
    	(2,), 
        'right',
	),	
    (
        'Test_l_1',
        get_einsum_dot_red_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_2',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'left',
	),
    (
        'Test_l_3',
        get_einsum_dot_red_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'left',
	),
    (
        'Test_l_4',
        get_einsum_dot_red_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'left',
	),
    (
        'Test_l_5',
        get_einsum_dot_red_string,
    	(5,), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_6',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(5,), 
        'left',
	),
    (
        'Test_r_1',
        get_einsum_dot_red_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_2',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'right',
	),
    (
        'Test_r_3',
        get_einsum_dot_red_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'right',
	),
    (
        'Test_r_4',
        get_einsum_dot_red_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'right',
	),
    (
        'Test_r_5',
        get_einsum_dot_red_string,
    	(2,), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_6',
        get_einsum_dot_red_string,
        (2, 3, 4, 5), 
    	(2,), 
        'right',
	),	
    (
        'Test_l_1',
        get_einsum_dot_exp_string,
    	(4, 5), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_2',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(4, 5), 
        'left',
	),
    (
        'Test_l_3',
        get_einsum_dot_exp_string,
    	(4, 1, 5), 
        (2, 3, 1, 4, 5), 
        'left',
	),
    (
        'Test_l_4',
        get_einsum_dot_exp_string,
        (2, 3, 1, 4, 5), 
    	(4, 1, 5), 
        'left',
	),
    (
        'Test_l_5',
        get_einsum_dot_exp_string,
    	(5,), 
        (2, 3, 4, 5), 
        'left',
	),
    (
        'Test_l_6',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(5,), 
        'left',
	),
    (
        'Test_r_1',
        get_einsum_dot_exp_string,
    	(2, 3), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_2',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2, 3), 
        'right',
	),
    (
        'Test_r_3',
        get_einsum_dot_exp_string,
    	(2, 1, 3), 
        (2, 3, 1, 4, 5), 
        'right',
	),
    (
        'Test_r_4',
        get_einsum_dot_exp_string,
        (2, 3, 1, 4, 5), 
    	(2, 1, 3), 
        'right',
	),
    (
        'Test_r_5',
        get_einsum_dot_exp_string,
    	(2,), 
        (2, 3, 4, 5), 
        'right',
	),
    (
        'Test_r_6',
        get_einsum_dot_exp_string,
        (2, 3, 4, 5), 
    	(2,), 
        'right',
	),	
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, method, x, y, side', data_test_failure)
def test_spark_config_success(name, method, x, y, side) -> None:
    """
        Einsum methods success validation.
    """
    fail = False
    try:
        get_einsum_dot_string(x, y, side=side)
    except:
        fail = True
    assert fail

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################