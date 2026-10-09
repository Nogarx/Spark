spark.core.addresses
====================

.. py:module:: spark.core.addresses


Attributes
----------

.. autoapisummary::

   spark.core.addresses.ANY_DEPTH
   spark.core.addresses.WILDCARDS


Functions
---------

.. autoapisummary::

   spark.core.addresses.is_pattern
   spark.core.addresses.matches
   spark.core.addresses.select


Module Contents
---------------

.. py:data:: ANY_DEPTH
   :value: '**'


   The name of a pattern standing for any number of names.

.. py:data:: WILDCARDS

   The characters making a name a pattern.

.. py:function:: is_pattern(address)

   Returns whether an address holds a wildcard.

   :param address: An address or a pattern.
   :type address: str

   :rtype: bool


.. py:function:: matches(pattern, address)

   Returns whether a pattern matches an address.

   :param pattern: A pattern, or an address, which matches itself only.
   :type pattern: str
   :param address: An address.
   :type address: str

   :returns: Whether every name of the pattern matches the name of the address in its place, and the
             port of the pattern the port of the address, when either has one.
   :rtype: bool


.. py:function:: select(pattern, addresses)

   Returns the addresses a pattern matches.

   :param pattern: A pattern, or an address.
   :type pattern: str
   :param addresses: The addresses to choose from.
   :type addresses: iterable of str

   :returns: The addresses matched, in the order given.
   :rtype: tuple of str


