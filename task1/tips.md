# Visual vs. Collision Geometry
You might notice that some of the provided objects include both visual and collision `<geom>` tags. This is a common and important practice in MuJoCo or other simulators when working with real-world assets like robot arms, furniture, or tools. 

In MuJoCo, collision detection is handled by computing contacts between geometries or mesh surfaces. For simple shapes like boxes, spheres, and capsules, this is fast and stable. However, for complex meshes, the collision engine attempts to resolve contacts across the entire volume of the mesh, which can lead to instability, jittering, or failed contacts. To address this, we provide a simplified collision mesh that approximates the object's shape with fewer faces or surfaces. These are defined in separate STL files or built-in geometries and are assigned a non-zero contype and conaffinity so they participate in physics. 

Meanwhile, the visual mesh is used purely for rendering and has both contype and conaffinity set to 0, ensuring it does not participate in collisions. 
This separation helps maintain simulation performance and stability while still preserving visual realism. You will see this useful later for RL (compilation)

