# ⚡ Spark: Modular Spiking Neural Networks

<div align="center"><img src="https://raw.githubusercontent.com/nogarx/Spark/main/images/spark_logo.png" alt="Spark Logo"></div>

<p align="center">
    <strong>
        Build, train, and deploy state-of-the-art Spiking Neural Networks with a powerful visual interface and JAX.
    </strong>
    <br/><br/>
    <a href="#">
        <img src="https://img.shields.io/badge/build-passing-brightgreen?style=for-the-badge" alt="Build Status">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/license-APACHE 2.0-blue?style=for-the-badge" alt="License">
    </a>
    <a href="#">
        <img src="https://img.shields.io/github/stars/nogarx/spark?style=for-the-badge" alt="GitHub Stars">
    </a>
    <a href="#">
        <img src="https://img.shields.io/pypi/v/spark?style=for-the-badge" alt="PyPI version">
    </a>
</p>

Spark is a next-generation framework designed to simplify and accelerate the research, development, and deployment of non-gradient-base Spiking Neural Networks (SNNs). Our goal is to make SNNs more accessible to researchers, engineers, and enthusiasts by abstracting away boilerplate code and providing intuitive tools for model creation and experimentation, while maintaing state-of-the-art performance.

## Key Features (Philosophy?)

⚡ <strong>High-Performance Backend:</strong> 
Powered by [JAX](https://jax.readthedocs.io/) and [Flax NNX](https://flax.readthedocs.io/), Spark enables just-in-time (JIT) compilation and state management of entire models.

🧠 <strong>Visual Model Builder:</strong> 
Design complex SNN architectures by dragging, dropping, and connecting pre-built neural components. 
No coding required for model design.

🧩 <strong>Modular & Extensible:</strong> 
Modular by construction.
Everything (that is worth interacting) in Spark is a self-contained module. 
Easily create, modify, and share custom neuron models, synapses, and learning rules.
Ever wanted a neuron with 3 Somas, 2 sets of Synapses and 2.5 Learning rules? As long as it can spike, you came to the right place! 

🔄 <strong>Seamless Workflow:</strong> 
Spiking neural networks are not special, why should they require special data?!. One of the core features of Spark is the concept of input and output interfaces which are simple modules that help you transform regular datasets into streams of spikes and transform streams of spikes into boring data formats like floats.

## Getting Started

Spark is available on PyPI, so it can be installed with:

```
pip install spark-snn
```

Or, for the latest development version, clone this repository:

```
git clone https://github.com/nogarx/spark.git
cd spark
pip install -e .
```

## The Spark Graph Editor

Design your network's structure, set parameters for each component, and connect them to create a model.

<div align="center">
    <img src="https://raw.githubusercontent.com/nogarx/Spark/main/images/spark_graph_editor.png" alt="Spark Logo"></div>
    <p>
        <em>The Spark visual interface for building SNNs.</em>
    </p>
</div>

Export your model to JSON and build it.

```
import spark

# Build model
model = spark.Brain.from_file('your_awesome_model.spark')
```

To harvest the true power of Spark your model needs to be JIT compiled. Jax requires your model to be traceable, which sometimes can be quite unintutive. Fortunately Flax NNX allows for entire model compilation thanks to its advance state managment. This drastically simplifies the procedure to write a simple function, in which arg_1, ..., arg_k are simply the name of your sources of the model.

```
import jax
import flax.nnx as nnx


@jax.jit
def run_model(graph, state, x_1, ..., x_k):
    # Reconstruct model
	model = nnx.merge(graph, state)
    # Compute
	out = model(arg_1=x, ..., arg_k=x_k)
    # Split the model and recover its state.
    _, state = nnx.split((model))
	return out, state
```

Finally, do the initial split of the model and use your function. 

```
import jax.numpy as jnp

# Some dummy data
x_1 = jnp.ones((64,), dtype=jnp.float16)
...
x_k = jnp.ones((128,), dtype=jnp.float16)

# Split the model
graph, state = nnx.split((model))

# Use run_model and reuse state!
out, state = run_model(graph, state, x, y)

# State now contains the updated state of the model!.
```
NOTE: The first time you call run_model it may take some time, depending on your model's complexity. This is normal as Jax is compiling your model.


## Roadmap

We have many exciting features planned. 

🔥 <strong>Components, a lot of them:</strong> 
Spark is built around the idea of modular neurons. Literature is full of really interesting ideas but integrating them to existing code is sometimes annoying and prone to errors. One of our goals is to transform those ideas into modular, reusable and plugable code.

📊 <strong>Built-in Visualization:</strong> 
(Maybe Coming Soon?) Tools for visualizing spike trains, membrane potentials, and network activity in real-time.

🧮 <strong>Surrogate gradients:</strong> 
Spark was build with the goal of building recurrent "Heabbian"-like learning schemes and as such it does not support surrogate gradients by default. Spark is also built on top of JAX, which makes automatic differentiation quite straight forward. However, surrogate gradients typically require a non-recurrent model to work properly which goes against the design philosophy of Spark. We are currently exploring how to integrate surrogate gradients in a way that does not violates our core design. 

## Contributing

Contributions are what make the open-source community such an amazing place. Any contributions you make are greatly appreciated.

## Citing Spark

You can use the following references to cite this repository, 

```
@article{spark_snn_github,
  author = {Mario Franco, Carlos Gershenson},
  title = {Spark: Modular Spiking Neural Networks},
  url = {},
  year = {2025},
}

@software{spark_snn_github,
  author = {Mario Franco, Carlos Gershenson},
  title = {Spark: Modular Spiking Neural Networks},
  url = {},
  version = {0.1.0},
  year = {2025},
}
```