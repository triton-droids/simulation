# Winter 2024 ADR 1

status: "accepted"
date: 2024-11-16
decision-makers: Jake, Anthony, Darin, Hrithik

Optional Elements:
consulted: Tech Advisor Tao
---

## Simulation Choice

### Context and Problem Statement

In order to run simulations, we need to run them on a single consistant simulator.

### Decision Drivers

* Easy to use
* Can be used on windows and mac
* Can run reinforcement learning, perferably on Python

### Considered Options

* Matlab
* Mujoco
* Isaac Sim
* Bullet Physics

### Decision Outcome

Chosen Option: Mujoco. We choose Mujoco because it was the only option that could run reinforcement learning, and run on both windows and mac. It is also one of the only options that allows for GPU parallelization, which will be useful alter. Our advisor Tao recommended Mujoco as well.