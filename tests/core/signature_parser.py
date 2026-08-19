
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

import pytest
from spark.core.signature_parser import is_instance, normalize_typehint

#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################

class DummyClass:
	pass

#-----------------------------------------------------------------------------------------------------------------------------------------------#

data_is_instance_test = [
    (
        'Simple_1',
        1, 
        (int,),
        True,
	),
    (
        'Simple_2',
        1.0, 
        (float,),
        True,
	),
    (
        'Simple_3',
        '1', 
        (str,),
        True,
	),
    (
        'Simple_4',
        1.0, 
        (int,),
        False,
	),
    (
        'Simple_5',
        '1', 
        (float,),
        False,
	),
    (
        'Simple_6',
        1, 
        (str,),
        False,
	),
    (
        'Tuple_1',
        (1,2,3), 
        (tuple,),
        True,
	),
    (
        'Tuple_2',
        (1,2,3), 
        (tuple[int, ...],),
        True,
	),
    (
        'Tuple_3',
        (1,2,3), 
        (tuple[int | float, ...],),
        True,
	),
    (
        'Tuple_4',
        (1,2.0,'3'), 
        (tuple[int, float, str],),
        True,
	),
    (
        'Tuple_5',
        (1,2,3), 
        (tuple[float, ...] | tuple[int, ...],),
        True,
	),
    (
        'Tuple_6',
        [1,2,3], 
        (tuple,),
        False,
	),
    (
        'Tuple_7',
        (1,2,3), 
        (tuple[int],),
        False,
	),
    (
        'Tuple_8',
        ('1','2','3'), 
        (tuple[int | float, ...],),
        False,
	),
	(
        'List_1',
        [1,2,3], 
        (list,),
        True,
	),
    (
        'List_2',
        [1,2,3], 
        (list[int],),
        True,
	),
	(
        'List_3',
        [1,2,3], 
        (list[float] | list[int],),
        True,
	),
	(
        'List_4',
        (1,2,3), 
        (list,),
        False,
	),
    (
        'List_5',
        [1.0,2.0,3.0], 
        (list[int],),
        False,
	),
	(
        'List_6',
        [1,2,3], 
        (list[float],),
        False,
	),
	(
        'Dict_1',
        {'1':1, '2':2, '3':3}, 
        (dict[str, int],),
        True,
	),
	(
        'Dict_2',
        {1:'1', 2:'2', 3:'3'}, 
        (dict[int, str],),
        True,
	),
	(
        'Dict_3',
        {'1':1, '2':2.0, '3':3}, 
        (dict[str, int | float],),
        True,
	),
	(
        'Dict_4',
        {'1':1, 2:2, '3':3}, 
        (dict[str | int, int],),
        True,
	),
	(
        'Dict_5',
        {1:1, 2:2, 3:3}, 
        (dict[str, int],),
        False,
	),
	(
        'Dict_6',
        {1:1, 2:2, 3:3}, 
        (dict[int, str],),
        False,
	),
	(
        'Dict_7',
        {1:[1,2,3], 2:[1,2,3], 3:[1,2,3]}, 
        (dict[int, list],),
        True,
	),
	(
        'Dict_8',
        {1:[1,2,3], 2:[1,2,3], 3:[1,2,3]}, 
        (dict[int, list[int]],),
        True,
	),
	(
        'Dict_9',
        {1:(1,2,3), 2:(1,2,3), 3:(1,2,3)}, 
        (dict[int, tuple[int,...]],),
        True,
	),
	(
        'Dict_10',
        {1:{'1':1.0}, 2:{'2':2.0}, 3:{'3':3.0}}, 
        (dict[int, dict[str, float]],),
        True,
	),
	(
        'Dict_11',
        {1:{'1':1.0}, 2:{'2':2.0}, 3:{'3':3.0}}, 
        (dict[int, dict[str, str]],),
        False,
	),
    (
        'Set_1',
        {1, 2, 3},
        (set),
        True,
	),
    (
        'Set_2',
        {1, 2, 3},
        (set[int]),
        True,
	),
    (
        'Set_2',
        {1, 2, 3},
        (set[float]),
        False,
	),
    (
        'Custom_Class_1',
		DummyClass(),
		(DummyClass,),
		True,
	),
    (
        'Custom_Class_2',
		(DummyClass(), DummyClass(), DummyClass(),),
		(tuple[DummyClass, ...],),
		True,
	),
    (
        'Custom_Class_3',
		[DummyClass(), DummyClass(), DummyClass(),],
		(list[DummyClass],),
		True,
	),
    (
        'Custom_Class_4',
		{'1':DummyClass(), '2':DummyClass(), '3':DummyClass(),},
		(dict[str, DummyClass],),
		True,
	),
    (
        'Custom_Class_5',
		{DummyClass(), DummyClass(), DummyClass()},
		(set[DummyClass],),
		True,
	),
    (
        'Mixed_1',
		(1,2,3),
		(tuple[int, ...], list[int], set[int],),
		True,
	),
    (
        'Mixed_2',
		[1,2,3],
		(tuple[int, ...], list[int], set[int],),
		True,
	),
    (
        'Mixed_3',
		{1,2,3},
		(tuple[int, ...], list[int], set[int],),
		True,
	),
    (
        'Mixed_4',
		(1,2,3),
		(tuple[int, ...] | list[int] | set[int],),
		True,
	),
    (
        'Mixed_5',
		[1,2,3],
		(tuple[int, ...] | list[int] | set[int],),
		True,
	),
    (
        'Mixed_6',
		{1,2,3},
		(tuple[int, ...] | list[int] | set[int],),
		True,
	),
    (
        'Mixed_7',
		([1,2,3],[1,2,3,],[1,2,3]),
		(tuple[list[int | float], ...] | list[list[int | float]] | set[int | float],),
		True,
	),
    (
        'Mixed_8',
		[[1,2,3],[1,2,3,],[1,2,3]],
		(tuple[list[int | float], ...] | list[list[int | float]] | set[int | float],),
		True,
	),
    (
        'Mixed_9',
		{1.0,2,'3'},
		(tuple[list[int | float], ...] | list[list[int | float]] | set[int | float | str],),
		True,
	),
    (
        'Mixed_10',
		([1,2,3],['1','2','3',],[1,2,3]),
		(tuple[list[int | float], ...] | list[list[int | float]] | set[int | float],),
		False,
	),
    (
        'Mixed_11',
		[[1,2,3],['1','2','3',],[1,2,3]],
		(tuple[list[int | float], ...] | list[list[int | float]] | set[int | float],),
		False,
	),
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, value, valid_types, expected', data_is_instance_test)
def test_is_instance(name, value, valid_types, expected) -> None:
	assert is_instance(value, valid_types) == expected
	
#-----------------------------------------------------------------------------------------------------------------------------------------------#

data_normalize_typehint_test = [
    (
        'Norm_1',
        int,
        (int,),
	),
	(
        'Norm_2',
        int | float,
        (int, float),
	),
	(
        'Norm_3',
        int | float | list,
        (int, float, list),
	),
	(
        'Norm_4',
        int | float | list,
        (int, float, list),
	),
	(
        'Norm_5',
        list[int] | list[float],
        (list[int], list[float]),
	),
	(
        'Norm_6',
        list[int] | list[list[int] | list[float]],
        (list[int], list[list[int]], list[list[float]]),
	),
	(
        'Norm_7',
        dict[str | int, float],
        (dict[str, float], dict[int, float]),
	),
	(
        'Norm_8',
        dict[str | int, float],
        (dict[str, float], dict[int, float]),
	),
	(
        'Norm_9',
        dict[str | int, DummyClass],
        (dict[str, DummyClass], dict[int, DummyClass]),
	),
	(
        'Norm_10',
        dict[str | int, list[DummyClass | int]],
        (dict[str, list[DummyClass]], dict[int, list[DummyClass]], dict[str, list[int]], dict[int, list[int]]),
	),
	(
        'Norm_10',
        int | dict[str | int, list[DummyClass | dict[int | str, DummyClass]]] | str,
        (int, dict[str, list[DummyClass]], dict[int, list[DummyClass]], dict[str, list[dict[int, DummyClass]]], 
		 dict[str, list[dict[str, DummyClass]]], dict[int, list[dict[int, DummyClass]]], dict[int, list[dict[str, DummyClass]]], str),
	),
]

#-----------------------------------------------------------------------------------------------------------------------------------------------#

@pytest.mark.parametrize('name, valid_types, expected', data_normalize_typehint_test)
def test_normalize_typehint(name, valid_types, expected) -> None:
	assert set(normalize_typehint(valid_types)) == set(expected)
	
#################################################################################################################################################
#-----------------------------------------------------------------------------------------------------------------------------------------------#
#################################################################################################################################################