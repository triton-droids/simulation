# Environment setup

## 1. Clone reponsitory branch
   ````bash
   git clone -b onboarding git@github.com:triton-droids/simulation.git

   cd simulation
   ````

## 2. Create and activate a new Conda environment
   ````bash
   conda create -n mujoco_env python==3.9.21

   conda activate mujoco_env
   ````

## 3. Install dependencies
  ````bash
  pip install -r requirements.txt
  ````

  >If you would like to install the standalone version of MuJoCo download the prebuilt binaries from the official [MuJoCo releases page](https://github.com/google-deepmind/mujoco/releases) and look up instructions for your machine, although this is not necessary for the following tasks. 
