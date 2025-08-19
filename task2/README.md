# Task 2: Imitation Learning
In this task, you will work with the simulation scene that you have created in task 1. The primary goal is to become familiar with MuJoCo’s API for accessing robot and environment state information and also begin to see how machine learning integrates with robotics control. 

We have provided the basic imitation learning pipeline (data collection → training → evaluation). You will fill in key pieces of the code to:
- Access and interpret MuJoCo state information (positions, quaternions, velocities, etc.)
- Collect demonstrations using the Gymnasium interface
- Train a neural network policy to imitate expert behavior
- Evaluate the trained policy in simulation

Along the way, you will gain experience with machine learning workflow for robotics, and start thinking about the challenges that come with it. 

By the end of this task, you should feel comfortable accessing and using simulation state information, navigating the Gymnasium interface, and reasoning about the role of machine learning in robotics control. These skills form the foundation for more advanced work in embodied AI.   

>*💡 Feel free to explore different approaches to data collection, neural network architectures, and more advanced algorithms and techniques for imitation learning.*

  <video controls style="width: 65%; height: auto; display: block; margin: 0 auto;">
    <source src="../assets/media/pick_and_place.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
