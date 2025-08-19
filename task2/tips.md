# Behavioral Cloning (BC) Overview



Behavioral cloning is a **supervised learning approach** for imitation learning.

In short: BC treats imitation as **a regression problem**, learning a mapping `s → a`.

---

## Challenges in Robotics

1. **Covariate shift / compounding errors**  
   - BC assumes the model sees states similar to the training data.  
   - Small errors during deployment may push the robot into **states not seen in demonstrations**, causing errors to compound.

2. **Multi-modality**  
   - Some tasks have multiple valid actions (e.g., multiple ways to grasp an object).  
   - Deterministic BC may average actions, producing invalid or suboptimal behavior.  
   - Using **stochastic policies** with Gaussian outputs can model multiple possibilities.

3. **Long-horizon tasks**  
   - Errors early in the trajectory can affect future states significantly.  


---