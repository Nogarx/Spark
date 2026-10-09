spark.recording.probe_targets
=============================

.. py:module:: spark.recording.probe_targets


Classes
-------

.. autoapisummary::

   spark.recording.probe_targets.ProbeTarget


Functions
---------

.. autoapisummary::

   spark.recording.probe_targets.get_probe_targets


Module Contents
---------------

.. py:class:: ProbeTarget

   Description of variable that a Probe can measure.

   Similar in spirit to Specs but for Probes.

   .. attribute:: address

      Probe address.

      :type: str

   .. attribute:: kind

      ``'port'`` or ``'attribute'``.

      :type: str

   .. attribute:: shape

      Shape of the value.

      :type: tuple of int

   .. attribute:: dtype

      Dtype of the recorded array. Bool for spike payloads.

      :type: numpy.dtype

   .. attribute:: payload

      Class name of the value of a port or property. ``'Variable'`` for a variable.

      :type: str

   .. attribute:: module

      Class name of the module producing the port or holding the attribute. For the inputs of
      a controller, the controller.

      :type: str

   .. seealso::

      :py:obj:`get_probe_targets`
          Lists every value of a controller that a probe can address.

      :py:obj:`Probe`
          A value read from a controller, and how it is recorded.


   .. py:attribute:: address
      :type:  str


   .. py:attribute:: kind
      :type:  str


   .. py:attribute:: shape
      :type:  tuple[int, ...]


   .. py:attribute:: dtype
      :type:  numpy.dtype


   .. py:attribute:: payload
      :type:  str


   .. py:attribute:: module
      :type:  str


   .. py:property:: size
      :type: int


      Number of entries of the value.


   .. py:property:: spikes
      :type: bool


      Whether the value is a spike payload.


   .. py:method:: __str__()


.. py:function:: get_probe_targets(controller, inputs)

   Returns a list of every variable (input, output or property), within the controller,
   that can be targeted using a Probe, as well as its shape and dtype.

   :param controller: The controller to be described.
   :type controller: Controller
   :param inputs: Input sample.
   :type inputs: dict[str, SparkPayload]

   :returns: Tuple of ProbeTargets pointing towards the controller's input/output/attributes,
             sorted by module path and name, that a Probe can target.
   :rtype: tuple of ProbeTarget

   :raises TypeError: When an input is not a payload.

   .. seealso::

      :py:obj:`ProbeTarget`
          A value of a controller that a probe can target.

      :py:obj:`Probe`
          A value read from a controller, and how it is recorded.

      :py:obj:`validate`
          Checks that every probe addresses something the controller produces.

   .. rubric:: Examples

   >>> for target in spark.recording.get_probe_targets(brain, inputs):
   ...     print(target)


