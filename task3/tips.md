# MuJoCo vs. MJX
MuJoCo is the physics engine that we will be using (for the time being) to experiment and validate our model and various control policies. MuJoCo is a lightweight engine that runs well on Colab. 

One challenge with using MuJoCo, however, is that it is built upon NumPy. NumPy runs only on CPU constraining computational power, i.e. the number of parallelizable environments during training, which is essential for reinforcement learning. 

As a result MJX allows MuJoCo to run on supported hardware. Please see supported platforms to ensure your machine supports JAX. MJX also runs very easily on Colab. 

Talk about JAX. what is JAX? Don’t get lost in the sauce, we can learn JAX later. What’s important now is learning to use MuJoCo / MJX for reinforcement learning and other ML techniques for robotics. 
