# How to find a stable starting pose

## Set up workflow

On two different terminals, load both the humanoid and scene xml files into the simulator

On terminal 1:

```bash
# Load the scene
python sim.py
```

On terminal 2:

```bash
# Load just the humanoid
python sim.py -h
```

## Workflow

1. Find a starting pose in the humanoid visualizer that you think is a good starting point. A good idea is to minimize the amount of joints that you would like to tweak in each iteration.

2. Copy the state into your desired keyframe in scene.xml. [sim.py](../sim.py) supports hot reload, so simply close the simulator and move to your desired keyframe to see the result. Repeat until the robot pose is stable.

## Tips

1. Try to maintain 2-3 keyframes at once, and use a "binary search" methodology to eventually close in on the values that will support the robot.

2. Once you need more reifined tweaking than what the simulator supports, try to isolate just one joint area and manually change the values in [scene.xml](../robot_description/scene.xml) until you get something good.
