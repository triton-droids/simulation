# Task 1: Create a Custom Simulation Task in MuJoCo

In this task, you will create a custom simulation scene for the MuJoCo physics engine using MJCF (MuJoCo XML) files. Your goal is to **compose a tabletop environment** featuring the **Arctos robot arm**, a **cube**, and a **bin**. The cube should be placed within reach of the robot arm, and the bin should be positioned such that it could be used as the target for a pick-and-place task. You do not need to implement any control or solve the task—this exercise focuses solely on scene composition.

<p align="center">
    <img src="../assets/media/panda_scene.png" alt="Humanoid robot" width="500"/>
</p>


You’ll be working with pre-existing MJCF object definitions, but feel free to explore or add your own to customize the environment. We’ve provided a [starter XML file](../assets/descriptions/DropCubeInBinEnv.xml) for you to construct the scene, which already includes a table asset as a starting point.

As part of the setup, you should also define a keyframe in your XML model titled `"home"`. This keyframe will set a default starting position for the robot’s joints. Use the following joint values for the home configuration:

`0 0.392699 0 -1.9635 0 2.35619 0.785398 0.04 0.04`


This ensures that the robot always begins in a consistent, neutral configuration, which will make later tasks involving control and learning more stable.

By the end of this task, you should feel comfortable with how MuJoCo scenes are structured, how to work with MJCF elements, and how to reason about object placement using coordinate frames and object geometry.

>You can find the necessary MJCF files for the arm and the provided objects [here](../assets/descriptions/).