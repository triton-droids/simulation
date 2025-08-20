# Task 3: Reinforcement Learning
In this task, you will being to explore reinforcement learning (RL) for robotic control. The goal is for the Panda robot arm to learn to pick up a cube and place it into the bin.

Unlike imitation learning, RL does not rely on expert demonstrations. Instead, the robot will learn by interacting with the environment, receiving feedback in the form of rewards. You will define the task setup, success conditions, and reward function, which guide the learning process.

You will implement the environment class in MJX, similar to task 2, to:
- Define initialization conditions for all objects and the robot 
- Specify success and failure conditions for the task, using the environment state at each time step
- Create a dense reward function that encourages the robot to pick up the cube and place it in the bin

Once your environment is complete, the provided training script will allow you to train a neural network policy using RL.

Along the way, you will gain experience with:
- Designing reward functions that guide learning and verify task solvability
- Understanding the role of environment initialization and randomization in RL


By the end of this task, you should feel confident in designing and implementing reinforcement learning environments, reasoning about reward shaping, and training policies to control a robot in simulation.

>💡 Feel free to experiment with different reward designs, object randomization strategies, and RL algorithms to improve learning efficiency and policy performance