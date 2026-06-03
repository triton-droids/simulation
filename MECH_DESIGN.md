# Mechanical Design Handoff

This guide describes how CAD should be structured so the robot can be exported cleanly into simulation and assembled consistently on hardware.

## Link Structure

Separate CAD into subassemblies by simulation links: each link should correspond to the rigid body between two joints. Avoid grouping multiple moving bodies into one exported part unless they are physically rigidly attached.

![Placeholder: CAD tree showing one subassembly per simulation link](docs/images/mech_design/01_link_subassemblies.png)

This makes it straightforward for the simulation developer to:

- Export one visual/collision asset per link.
- Attach joints between links.
- Assign mass, center of mass, inertia, and material properties per link.

## World Origin

The preferred CAD world origin is:

- On the floor plane at the feet contact height.
- Centered between the feet in the robot's nominal standing pose.
- Centered on the robot in the `xy` plane.

![Placeholder: full humanoid CAD standing pose with world origin at floor center](docs/images/mech_design/02_world_origin_floor_center.png)

Using this convention makes all CAD exports share a common reference and makes it easier to define simulation points such as feet, joints, IMU frames, and body offsets.

## IMU Reference

Any IMU mounting frame or reference point should be measured accurately relative to the CAD world origin. Confirm the final IMU location and orientation with the electrical and software teams before freezing the design.

![Placeholder: IMU mounting location with position and orientation relative to CAD origin](docs/images/mech_design/03_imu_reference.png)

Record:

- IMU position relative to the world origin.
- IMU orientation relative to the robot frame.
- Any mounting offsets, adapters, or expected assembly tolerances.

## Materials

Assign realistic material information to each body, especially contact surfaces such as feet. Foot material should be chosen and documented so simulation can use appropriate friction and restitution values.

![Placeholder: CAD material assignment view highlighting foot sole/contact material](docs/images/mech_design/04_materials_and_friction.png)

At minimum, document:

- Body/link material.
- Foot sole/contact material.
- Any coatings, pads, or replaceable contact surfaces.

## Mass Properties

URDF generation requires per-link mass properties. For each simulation link, provide:

![Placeholder: CAD mass properties dialog showing mass and center of mass for one link](docs/images/mech_design/05_link_mass_properties.png)

- Mass.
- Center of mass.
- Moment of inertia tensor.
- The coordinate frame used for the inertia values.

Ideally, inertia should be reported about the link's own reference frame, commonly located at or aligned with the link center of mass. If the CAD tool exports inertia about another point, clearly document that frame so it can be transformed correctly for URDF.

![Placeholder: CAD inertia tensor export for one link, including reference frame](docs/images/mech_design/06_moment_of_inertia.png)

## Joint Measurements

Provide accurate joint locations measured from the CAD world origin. Each joint should include:

![Placeholder: dimensioned CAD view from world origin to each joint center](docs/images/mech_design/07_joint_distance_measurements.png)

- Joint position relative to the world origin.
- Joint axis direction.
- Parent link and child link.
- Nominal joint angle in the standing pose.
- Any mechanical offset required for the robot to stand naturally.

When possible, design joints so their axes align with the world frame axes in the nominal pose. This makes joint definition in simulation much less error-prone.

![Placeholder: joint axis visualization aligned to world frame axes](docs/images/mech_design/08_joint_axis_alignment.png)

If a joint needs an intentional displacement or angular offset, measure and document it explicitly so simulation can reproduce the real robot's natural standing configuration.

![Placeholder: natural standing pose showing joint offsets from mechanical zero](docs/images/mech_design/09_natural_standing_offsets.png)

## Fixturing And Zeroing

Include mechanical designs or procedures for fixturing the robot so motors can be zeroed repeatably on the real hardware.

![Placeholder: zeroing fixture design with constrained links and reference stops](docs/images/mech_design/10_zeroing_fixture.png)

The fixture documentation should define:

- The physical zero pose.
- How each link is constrained during zeroing.
- Which surfaces, pins, or hard stops are used as references.
- Any special tools required.
- Expected tolerance or repeatability.

This is separate from motor configuration and identification, which should be documented independently.

## Photo Checklist

Procure or export the following images and place them under `docs/images/mech_design/` using the filenames shown above:

1. `01_link_subassemblies.png`: CAD tree or exploded view showing one subassembly per simulation link.
2. `02_world_origin_floor_center.png`: full standing robot with world origin at floor center between the feet.
3. `03_imu_reference.png`: IMU mount position and orientation, dimensioned relative to the CAD world origin.
4. `04_materials_and_friction.png`: material assignment view highlighting body materials and foot contact material.
5. `05_link_mass_properties.png`: mass and center-of-mass properties for a representative link.
6. `06_moment_of_inertia.png`: inertia tensor export for a representative link, including the reference frame.
7. `07_joint_distance_measurements.png`: dimensioned view from world origin to each joint center.
8. `08_joint_axis_alignment.png`: joint axis visualization showing alignment with world axes.
9. `09_natural_standing_offsets.png`: natural standing pose with any joint offsets from mechanical zero called out.
10. `10_zeroing_fixture.png`: fixture design or procedure image for repeatable motor zeroing.
