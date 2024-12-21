# Learning MuJoCo
MuJoCo (Multi-Joint dynamics with Contact) is a powerful physics engine used for simulating rigid body dynamics. It aims to facilitate research and development in robotics and other areas where fast and accurate simulation is needed. MuJoCo is a C/C++ library that comes standard with native python bindings.

This branch provides resources and guidance to help you get started.

## Getting Started
- **Beginner's Guide**: 
A general beginners guide and MuJoCo tutorial using Python bindings is available [here](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb).
- **Interediate Guide**: An intermediate tutorial on XML modeling and MuJoCo environments can be found [here](./learning_mujoco.ipynb).
- **MuJoCo Bootcamp**: A bootcamp covering implementation and designof control systems, state machines, kinematics, and more is available [here](https://pab47.github.io/mujoco.html). 



## Useful Resources / Links
1. **C Implementation of Key Structs in MuJoCo**

Access the [C implementation](https://github.com/google-deepmind/mujoco/blob/main/include/mujoco/mjmodel.h) of crucial structs like `mjmodel` and `mjdata` for interacting with MuJoCo at a low level. 

2. **MuJoCo API Reference**

Explore the [API Reference](https://mujoco.readthedocs.io/en/stable/APIreference/index.html) for Python bindings. These are in sync with the header files above.

3. **MJCF (MuJoCo XML) Reference**

Here is the [XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html) page to learn about XML elements and properties for creating models 

<br>

Everything provided can be found on MuJoCo's [main site](https://mujoco.readthedocs.io/en/stable/overview.html).



