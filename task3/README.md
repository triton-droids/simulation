# Task 3: Reinforcement Learning
In this task, you’ll use reinforcement learning (RL) to teach a Panda robot arm to pick up a cube and place it into a bin. Unlike imitation learning, RL learns from interaction and reward signals, no expert demos required.

You will implement the environment class using MJX, similar to Task 2. For some information on MJX, see the official [MJX documentation](https://mujoco.readthedocs.io/en/stable/mjx.html).

Your environment should:
- Initialize the scene (robot, cube, bin) and any randomization ranges.
- Define success/failure conditions using the state at each step.
- Provide a dense reward that encourages (i) reaching, (ii) grasping/lifting, and (iii) transporting/placing.
>Keep it simple first. You can expand later.




By the end of this task, you should feel confident in designing and implementing reinforcement learning environments, reasoning about reward shaping, and training policies to control a robot in simulation.

>💡 Feel free to experiment with different reward designs, object randomization strategies, and RL algorithms to improve learning efficiency and policy performance

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/triton-droids/simulation/blob/onboarding/task3/reinforcement_learning.ipynb?copy=true)
