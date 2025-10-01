# Environment setup

## 1. Clone reponsitory branch
   ````bash
   git clone -b onboarding git@github.com:triton-droids/simulation.git

   cd simulation
   ````

## 2. Create and activate a new Conda environment
   ````bash
   conda env create -f environment.yml

   conda activate mujoco_cpu
   ````

  >If you would like to install the standalone version of MuJoCo download the prebuilt binaries from the official [MuJoCo releases page](https://github.com/google-deepmind/mujoco/releases) and look up instructions for your machine, although this is not necessary for the following tasks. 
