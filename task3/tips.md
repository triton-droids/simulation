# MuJoCo vs. MJX
MuJoCo is the physics engine we use to simulate and validate our models and control policies. It’s lightweight, efficient, and runs well on Colab, making it ideal for experimentation.

However, MuJoCo is built on top of NumPy, which only runs on the CPU. This limits the computational power available during training, especially when you want to run many parallel environments.

MJX solves this problem by integrating MuJoCo with JAX, a numerical computing library that supports GPU and TPU acceleration. This allows simulations and training to run much faster on supported hardware. You don’t need to worry about JAX too much right now. What matters for us is that MJX lets us run MuJoCo efficiently, we will use exisitng jax algorithms to train our model.

Before you start, make sure your machine [supports JAX](https://docs.jax.dev/en/latest/installation.html#supported-platforms) if you plan to run simulations locally. Otherwise, Colab provides an easy way to get started with GPU-accelerated MJX.