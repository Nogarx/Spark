spark.core.utils
================

.. py:module:: spark.core.utils


Classes
-------

.. autoapisummary::

   spark.core.utils.TwoKeyDict


Functions
---------

.. autoapisummary::

   spark.core.utils.normalize_str
   spark.core.utils.to_human_readable
   spark.core.utils.get_einsum_labels
   spark.core.utils.get_axes_einsum_labels
   spark.core.utils.get_einsum_dot_string
   spark.core.utils.get_einsum_dot_red_string
   spark.core.utils.get_einsum_dot_exp_string
   spark.core.utils.validate_shape
   spark.core.utils.validate_list_shape
   spark.core.utils.merge_shape_list
   spark.core.utils.contract_axes
   spark.core.utils.contracted_shape
   spark.core.utils.is_shape
   spark.core.utils.is_list_shape
   spark.core.utils.is_dict_of
   spark.core.utils.is_list_of
   spark.core.utils.is_dtype
   spark.core.utils.is_float
   spark.core.utils.ascii_tree


Module Contents
---------------

.. py:function:: normalize_str(s)

   Converts a string to lowercase snake_case.

   :param s: String to normalize.
   :type s: str

   :returns: The normalized string.
   :rtype: str


.. py:function:: to_human_readable(s, capitalize_all = True)

   Converts a string from a programming case into a readable one.

   :param s: String to convert.
   :type s: str
   :param capitalize_all: Title-case every word instead of the first one alone.
   :type capitalize_all: bool, default False

   :returns: The readable string.
   :rtype: str


.. py:function:: get_einsum_labels(num_dims, offset = 0)

   Builds a run of einsum labels.

   :param num_dims: Number of labels to generate.
   :type num_dims: int
   :param offset: Number of labels to skip before the first one.
   :type offset: int, default 0

   :returns: ``num_dims`` distinct labels, starting after ``offset``.
   :rtype: str


.. py:function:: get_axes_einsum_labels(axes, ignore_repeated = False)

   Builds the einsum labels naming a given set of axes.

   :param axes: Axis indices to name.
   :type axes: tuple of int
   :param ignore_repeated: Allow the same axis to appear more than once.
   :type ignore_repeated: bool, default False

   :returns: One label per entry of ``axes``, in the order given.
   :rtype: str

   :raises ValueError: If an axis is negative, beyond the number of available labels, or repeated while
       ``ignore_repeated`` is False.


.. py:function:: get_einsum_dot_string(x, y, ignore_one_dims = True, side = 'right')

   Builds the einsum string of a generalized dot product.

   The shared axes are contracted and the remaining ones are kept::

       right:  (c,d)*(a,b,c,d)=(a,b)  cd,abcd->ab    (a,b,c,d)*(c,d)=(a,b)  abcd,cd->ab
       left:   (a,b)*(a,b,c,d)=(c,d)  ab,abcd->cd    (a,b,c,d)*(a,b)=(c,d)  abcd,ab->cd

   :param x: Shape of the first operand.
   :type x: tuple of int
   :param y: Shape of the second operand.
   :type y: tuple of int
   :param ignore_one_dims: Drop the axes of length one before building the labels.
   :type ignore_one_dims: bool, default True
   :param side: Which end of the longer shape the shorter one aligns with.
   :type side: {'right', 'left'}, default 'right'

   :returns: The einsum string.
   :rtype: str


.. py:function:: get_einsum_dot_red_string(x, y, ignore_one_dims = True, side = 'right')

   Builds the einsum string of a generalized dot product that keeps the shared axes.

   The shared axes are reduced onto themselves rather than contracted away::

       right:  (a,b)*(a,b,c,d)=(a,b)  ab,abcd->ab    (a,b,c,d)*(a,b)=(a,b)  abcd,ab->ab
       left:   (c,d)*(a,b,c,d)=(c,d)  cd,abcd->cd    (a,b,c,d)*(c,d)=(c,d)  abcd,cd->cd

   :param x: Shape of the first operand.
   :type x: tuple of int
   :param y: Shape of the second operand.
   :type y: tuple of int
   :param ignore_one_dims: Drop the axes of length one before building the labels.
   :type ignore_one_dims: bool, default True
   :param side: Which end of the longer shape the shorter one aligns with.
   :type side: {'right', 'left'}, default 'right'

   :returns: The einsum string.
   :rtype: str


.. py:function:: get_einsum_dot_exp_string(x, y, ignore_one_dims = True, side = 'right')

   Builds the einsum string of a generalized outer product.

   Nothing is contracted; the result carries the axes of both operands::

       right:  (a,b)*(a,b,c,d)=(a,b,c,d)  ab,abcd->abcd
       left:   (c,d)*(a,b,c,d)=(a,b,c,d)  cd,abcd->abcd
       none:   (a,b)*(c,d)=(a,b,c,d)      ab,cd->abcd
       flip:   (a,b)*(c,d)=(c,d,a,b)      cd,ab->abcd

   :param x: Shape of the first operand.
   :type x: tuple of int
   :param y: Shape of the second operand.
   :type y: tuple of int
   :param ignore_one_dims: Drop the axes of length one before building the labels.
   :type ignore_one_dims: bool, default True
   :param side: How the axes of the two operands are laid out in the result.
   :type side: {'right', 'left', 'none', 'flip'}, default 'right'

   :returns: The einsum string.
   :rtype: str


.. py:function:: validate_shape(obj)

   Checks that an object reads as a shape and returns it.

   :param obj: Object to read as a shape.
   :type obj: Any

   :returns: The shape.
   :rtype: tuple of int

   :raises TypeError: If the object does not read as a tuple of integers.


.. py:function:: validate_list_shape(obj)

   Checks that an object reads as a list of shapes and returns it.

   :param obj: Object to read as a list of shapes.
   :type obj: Any

   :returns: The shapes.
   :rtype: list of tuple of int

   :raises TypeError: If the object does not read as a list of tuples of integers.


.. py:function:: merge_shape_list(shape_list)

   Merges a list of shapes into one.

   :param shape_list: Shapes to merge.
   :type shape_list: list of tuple of int

   :returns: The merged shape.
   :rtype: tuple of int


.. py:function:: contract_axes(array, axes, shape)

   Contracts an array over the given axes when it is constant along them.

   :param array: Array to contract.
   :type array: Any
   :param axes: Axes to contract over.
   :type axes: tuple of int
   :param shape: Shape the array is read against.
   :type shape: tuple of int

   :returns: * **array** (*Any*) -- The contracted array, or the original one when the contraction does not apply.
             * **contracted** (*bool*) -- Whether the array was contracted.


.. py:function:: contracted_shape(shape, axes)

   Returns the shape `contract_axes` produces.

   :param shape: Shape before the contraction.
   :type shape: tuple of int
   :param axes: Axes contracted over.
   :type axes: tuple of int

   :returns: The shape with the contracted axes dropped.
   :rtype: tuple of int


.. py:function:: is_shape(obj)

   Whether an object reads as a shape.

   :param obj: Object to check.
   :type obj: Any

   :rtype: bool


.. py:function:: is_list_shape(obj)

   Whether an object reads as a list of shapes.

   :param obj: Object to check.
   :type obj: Any

   :rtype: bool


.. py:function:: is_dict_of(obj, value_cls, key_cls = str)

   Whether an object is a ``dict[key_cls, value_cls]``.

   :param obj: Object to check.
   :type obj: Any
   :param value_cls: Class the values are checked against.
   :type value_cls: type
   :param key_cls: Class the keys are checked against.
   :type key_cls: type, default str

   :rtype: bool


.. py:function:: is_list_of(obj, cls)

   Whether an object is a ``list[cls]``.

   :param obj: Object to check.
   :type obj: Any
   :param cls: Class the entries are checked against.
   :type cls: type

   :rtype: bool


.. py:function:: is_dtype(obj)

   Whether an object is a dtype.

   :param obj: Object to check.
   :type obj: Any

   :rtype: bool


.. py:function:: is_float(obj)

   Whether an object is a floating point dtype.

   :param obj: Object to check.
   :type obj: Any

   :returns: True for a real floating dtype, False for anything else, including integer and
             boolean dtypes.
   :rtype: bool


.. py:function:: ascii_tree(text)

   Renders indented text as an ASCII tree.

   :param text: Lines whose depth is given by their leading spaces.
   :type text: str

   :returns: The tree.
   :rtype: str


.. py:class:: TwoKeyDict(data = None)

   Bases: :py:obj:`collections.abc.MutableMapping`\ [\ :py:obj:`tuple`\ [\ :py:obj:`_K1`\ , :py:obj:`_K2`\ ]\ , :py:obj:`_VT`\ ], :py:obj:`Generic`\ [\ :py:obj:`_K1`\ , :py:obj:`_K2`\ , :py:obj:`_VT`\ ]


   Mapping addressed by a pair of keys.

   ``d[k1, k2]`` reads one entry and ``d[k1]`` reads the inner mapping under the first key.
   Used for the values a graph addresses by (module name, port name).

   :param data: Initial contents, as nested mappings.
   :type data: dict of K1 to dict of K2 to VT, optional


   .. py:method:: __getitem__(keys: tuple[_K1, _K2]) -> _VT
                  __getitem__(keys: _K1) -> dict[_K2, _VT]


   .. py:method:: __setitem__(keys: _K1, value: dict[_K2, _VT]) -> None
                  __setitem__(keys: tuple[_K1, _K2], value: _VT) -> None


   .. py:method:: __delitem__(keys: _K1) -> None
                  __delitem__(keys: tuple[_K1, _K2]) -> None


   .. py:method:: __len__()


   .. py:method:: __iter__()


   .. py:method:: __str__()


   .. py:method:: __repr__()


   .. py:method:: __contains__(keys: _K1) -> bool
                  __contains__(keys: tuple[_K1, _K2]) -> bool


   .. py:method:: keys()

      D.keys() -> a set-like object providing a view on D's keys



   .. py:method:: values()

      D.values() -> an object providing a view on D's values



   .. py:method:: items()

      D.items() -> a set-like object providing a view on D's items



   .. py:method:: tree_flatten()


   .. py:method:: tree_flatten_with_keys()


   .. py:method:: tree_unflatten(aux_data, children)
      :classmethod:



