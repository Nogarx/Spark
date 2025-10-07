.. Spark documentation master file, created by
   sphinx-quickstart on Tue Oct  7 11:58:17 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Spark documentation
===================

Spark is a next-generation framework designed to simplify and accelerate the research, development, and deployment of non-gradient-base Spiking Neural Networks (SNNs). Our goal is to make SNNs more accessible to researchers, engineers, and enthusiasts by abstracting away boilerplate code and providing intuitive tools for model creation and experimentation, while maintaing state-of-the-art performance.

## Key Features (Philosophy?)

⚡ <strong>High-Performance Backend:</strong> 
Powered by [JAX](https://jax.readthedocs.io/) and [Flax NNX](https://flax.readthedocs.io/), Spark enables just-in-time (JIT) compilation and state management of entire models.

🧩 <strong>Modular & Extensible:</strong> 
Modular by construction.
Everything (that is worth interacting) in Spark is a self-contained module. 
Easily create, modify, and share custom neuron models, synapses, and learning rules.
Ever wanted a neuron with 3 Somas, 2 sets of Synapses and 2.5 Learning rules? As long as it can spike, you came to the right place! 

🔄 <strong>Seamless Workflow:</strong> 
Spiking neural networks are not special, why should they require special data?!. One of the core features of Spark is the concept of input and output interfaces which are simple modules that help you transform regular datasets into streams of spikes and transform streams of spikes back into boring data formats like floats.

🧠 <strong>Graph Editor:</strong> 
Design complex SNN architectures by dragging, dropping, and connecting pre-built neural components. 
No coding required for model design.

Contents
--------

.. toctree::
   :maxdepth: 5
   
   autoapi/index
