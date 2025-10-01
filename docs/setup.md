# Environment setup

note this assumes you have conda installed already for dependency management.

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

## 3. Upload conda environment for jupyter notebook usage
   ````bash
   python -m ipykernel install --user --name mujoco_cpu --display-name "Python (mujoco_cpu)"
   ````

## 4. Workflow (Enter each task directory on your terminal and run the following command to open )
   ````bash
   jupyter lab
   ````

  >If you would like to install the standalone version of MuJoCo download the prebuilt binaries from the official [MuJoCo releases page](https://github.com/google-deepmind/mujoco/releases) and look up instructions for your machine, although this is not necessary for the following tasks. 
