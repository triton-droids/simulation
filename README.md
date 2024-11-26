Refer to https://docs.google.com/document/d/1hz5qZrm4ITF_1TlYCzt-1vlaxFZolGrV2-ytaYGXRvM for guide on how to use MuJoCo

This repository contains everything simulations related.

admin/adrs:

folder containing architectural decision records (ADR)


pysims: 

Folder containing Python scripts to run MuJoCo simulations:
  - `models` folder:
    - Contains the XML models to use in simulation
  - `mujoco_run.py`:
    - The main file that tests sensors/actuators on the default `humanoid.xml`


