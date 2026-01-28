spark.core.serializer
=====================

.. py:module:: spark.core.serializer


Attributes
----------

.. autoapisummary::

   spark.core.serializer.T


Classes
-------

.. autoapisummary::

   spark.core.serializer.SparkJSONEncoder
   spark.core.serializer.SparkJSONDecoder


Module Contents
---------------

.. py:data:: T

.. py:class:: SparkJSONEncoder(*args, is_partial = False, **kwargs)

   Bases: :py:obj:`json.JSONEncoder`


   Custom JSON encoder to handle common types encounter in Spark.

   Constructor for JSONEncoder, with sensible defaults.

   If skipkeys is false, then it is a TypeError to attempt
   encoding of keys that are not str, int, float, bool or None.
   If skipkeys is True, such items are simply skipped.

   If ensure_ascii is true, the output is guaranteed to be str objects
   with all incoming non-ASCII and non-printable characters escaped.
   If ensure_ascii is false, the output can contain non-ASCII and
   non-printable characters.

   If check_circular is true, then lists, dicts, and custom encoded
   objects will be checked for circular references during encoding to
   prevent an infinite recursion (which would cause an RecursionError).
   Otherwise, no such check takes place.

   If allow_nan is true, then NaN, Infinity, and -Infinity will be
   encoded as such.  This behavior is not JSON specification compliant,
   but is consistent with most JavaScript based encoders and decoders.
   Otherwise, it will be a ValueError to encode such floats.

   If sort_keys is true, then the output of dictionaries will be
   sorted by key; this is useful for regression tests to ensure
   that JSON serializations can be compared on a day-to-day basis.

   If indent is a non-negative integer, then JSON array
   elements and object members will be pretty-printed with that
   indent level.  An indent level of 0 will only insert newlines.
   None is the most compact representation.

   If specified, separators should be an (item_separator,
   key_separator) tuple.  The default is (', ', ': ') if *indent* is
   ``None`` and (',', ': ') otherwise.  To get the most compact JSON
   representation, you should specify (',', ':') to eliminate
   whitespace.

   If specified, default is a function that gets called for objects
   that can't otherwise be serialized.  It should return a JSON
   encodable version of the object or raise a ``TypeError``.



   .. py:attribute:: __version__
      :value: '1.0'



   .. py:method:: encode(obj)

      Return a JSON string representation of a Python data structure.

      >>> from json.encoder import JSONEncoder
      >>> JSONEncoder().encode({"foo": ["bar", "baz"]})
      '{"foo": ["bar", "baz"]}'




   .. py:method:: default(obj)

      Implement this method in a subclass such that it returns
      a serializable object for ``o``, or calls the base implementation
      (to raise a ``TypeError``).

      For example, to support arbitrary iterators, you could
      implement default like this::

          def default(self, o):
              try:
                  iterable = iter(o)
              except TypeError:
                  pass
              else:
                  return list(iterable)
              # Let the base class default method raise the TypeError
              return super().default(o)




.. py:class:: SparkJSONDecoder(*args, ignore_version = False, is_partial = False, **kwargs)

   Bases: :py:obj:`json.JSONDecoder`


   Custom JSON decoder to handle common types encounter in Spark.

   ``object_hook``, if specified, will be called with the result
   of every JSON object decoded and its return value will be used in
   place of the given ``dict``.  This can be used to provide custom
   deserializations (e.g. to support JSON-RPC class hinting).

   ``object_pairs_hook``, if specified will be called with the result
   of every JSON object decoded with an ordered list of pairs.  The
   return value of ``object_pairs_hook`` will be used instead of the
   ``dict``.  This feature can be used to implement custom decoders.
   If ``object_hook`` is also defined, the ``object_pairs_hook`` takes
   priority.

   ``parse_float``, if specified, will be called with the string
   of every JSON float to be decoded. By default this is equivalent to
   float(num_str). This can be used to use another datatype or parser
   for JSON floats (e.g. decimal.Decimal).

   ``parse_int``, if specified, will be called with the string
   of every JSON int to be decoded. By default this is equivalent to
   int(num_str). This can be used to use another datatype or parser
   for JSON integers (e.g. float).

   ``parse_constant``, if specified, will be called with one of the
   following strings: -Infinity, Infinity, NaN.
   This can be used to raise an exception if invalid JSON numbers
   are encountered.

   If ``strict`` is false (true is the default), then control
   characters will be allowed inside strings.  Control characters in
   this context are those with character codes in the 0-31 range,
   including ``'\t'`` (tab), ``'\n'``, ``'\r'`` and ``'\0'``.


   .. py:attribute:: __supported_versions__


   .. py:method:: object_hook(obj)


