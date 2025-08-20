# 1. MuJoCo Installation Guide (Python Bindings)

## Prerequisites

Before installing MuJoCo, make sure you have:

- **Conda** (from [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- **Python 3.8+**

---

## Installation via Conda Environment (Recommended)

The easiest way to install is inside a dedicated Conda environment with pip.

1. Create a new Conda environment (choose Python version 3.10 or newer):

   ```bash
   conda create -n mujoco_env python
   ```
2. Activate the environment:
    ```bash
   conda activate mujoco_env
   ```

3. Install MuJoCo with pip:
    ```bash
    pip install mujoco
    ```

4. Verify installation:
    ```bash
    import mujoco
    print(mujoco.__version__)
    ```

>If you would like to install the standalone version of MuJoCo download the prebuilt binaries from the official [MuJoCo releases page](https://github.com/google-deepmind/mujoco/releases) and look up instructions for your machine, although this is not necessary for the following tasks. 

---

# Environment setup

## Clone reponsitory branch
   ````bash
   git clone -b onboarding git@github.com:triton-droids/simulation.git
   ````

## Install dependencies
  ````bash
  conda activate mujoco_env
  pip install -r requirements.txt
  ````
