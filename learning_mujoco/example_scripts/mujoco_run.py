import time
import numpy as np
import mujoco
import mujoco.viewer

# Load the model and data
m = mujoco.MjModel.from_xml_path('/Users/darin/desktop/_/sims/pysims/models/humanoid.xml')
d = mujoco.MjData(m)

#############################################
###### Set model to a position based on keyframe in .xml file
#############################################


def load_keyframe(model, data, keyframe_name):
    # Get the keyframe ID by name
    keyframe_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    
    # Load the keyframe's qpos and qvel into the current state
    data.qpos = model.key_qpos[keyframe_id]
    data.qvel = model.key_qvel[keyframe_id]

load_keyframe(m, d, "squat")


#############################################
#############################################
#############################################

# Control indices for leg joints
right_leg_joints = [3, 4, 5, 6, 7, 8]
left_leg_joints = [9, 10, 11, 12, 13, 14]

# Define gait and control parameters
step_duration = 0.5

# Flag to control pausing
paused = False
stop_knee = False

# Function to toggle pause
def toggle_pause():
    global paused
    paused = not paused
    print(f"Paused: {paused}")

amplitude = 0.4
kp = 100  # Proportional gain for balance control
kd = 5    # Derivative gain for balance control

# Initialize the state machine variables
time_in_phase = 0
left_leg_swing = True  # Start with left leg in the swing phase

def balance_control():
    """Apply PD control to keep the torso upright."""
    # Get the torso's orientation (body id 1 is assumed to be the torso)
    torso_tilt = d.xquat[1, 1]  # Example: pitch of torso
    torso_angular_vel = d.qvel[1]  # Angular velocity around the pitch axis
    
    # PD control to apply counter-torque based on tilt
    balance_torque = -kp * torso_tilt - kd * torso_angular_vel
    d.ctrl[0] = balance_torque  # Apply to abdomen_z (or adjust based on the correct torso control)
    
def move_legs():
    # Apply control based on the current phase
    if left_leg_swing:
        # Left leg in swing phase, right leg in stance phase
        for i in left_leg_joints:
            d.ctrl[i] = amplitude * np.sin(2 * np.pi * time_in_phase / step_duration)
        for i in right_leg_joints:
            d.ctrl[i] = 0  # Keep right leg steady in stance
    else:
        # Right leg in swing phase, left leg in stance phase
        for i in right_leg_joints:
            d.ctrl[i] = amplitude * np.sin(2 * np.pi * time_in_phase / step_duration)
        for i in left_leg_joints:
            d.ctrl[i] = 0  # Keep left leg steady in stance
            
def stand_up_from_squat():
    # Indices for actuators controlling the legs
    hip_actuators = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_y_left"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "hip_y_right"),
    ]
    knee_actuators = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_left"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "knee_right"),
    ]
    
    ankle_actuators = [
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_y_left"),
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "ankle_y_right"),
    ]
    
    abdomen_y_actuator = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "abdomen_y")
    
    # Set target controls for standing up
    if time.time() - start > 1 and not stop_knee:
        for knee_actuator in knee_actuators:
            d.ctrl[knee_actuator] = 0.05  # Extend knees
    elif time.time() - start <= 1 and not stop_knee:
        for knee_actuator in knee_actuators:
            d.ctrl[knee_actuator] = 0.5  # Extend knees
    else:
        for knee_actuator in knee_actuators:
            d.ctrl[knee_actuator] = 0  # Stop extending knees if stop_knee is true
        
    if time.time() - start > 0.1:
        for hip_actuator in hip_actuators:
            d.ctrl[hip_actuator] = 0.2  # Extend hips
            
        for ankle_act in ankle_actuators:
            d.ctrl[ankle_act] = 0.05
            
        d.ctrl[abdomen_y_actuator] = 0.4
        
    # Get the vertical position of the torso
    torso_z = d.qpos[2]  # Assuming the 3rd element is the Z position of the torso

    # Check if the humanoid is upright
    if torso_z > 1.11:  # Adjust threshold as needed
        print("Humanoid is upright!")
        toggle_pause()
        return

            
def test_sensors():
    # Get the indices of the sensors
    right_foot_force_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_force")
    right_foot_torque_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "right_foot_torque")
    left_foot_force_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_force")
    left_foot_torque_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "left_foot_torque")
    knee_right_sensor_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "knee_right_pos")
    knee_left_sensor_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "knee_left_pos")
    
    # Access the SubtreeCoM sensor index by name
    com_sensor_idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "center_of_mass")

    # Retrieve sensor data
    right_foot_force = d.sensordata[right_foot_force_idx:right_foot_force_idx + 3]  # Fx, Fy, Fz
    right_foot_torque = d.sensordata[right_foot_torque_idx:right_foot_torque_idx + 3]  # Tx, Ty, Tz

    left_foot_force = d.sensordata[left_foot_force_idx:left_foot_force_idx + 3]  # Fx, Fy, Fz
    left_foot_torque = d.sensordata[left_foot_torque_idx:left_foot_torque_idx + 3]  # Tx, Ty, Tz
    
    # Retrieve the CoM data from the sensordata array
    robot_com = d.sensordata[com_sensor_idx:com_sensor_idx + 3]
    
    # Retrieve knee joint position
    knee_right_pos = d.sensordata[knee_right_sensor_idx]
    knee_left_pos = d.sensordata[knee_left_sensor_idx]
    
    # if knee_left_pos >  


    # Print the knee angles
    print(f"Knee Right Position: {knee_right_pos}, Knee Left Position: {knee_left_pos}")
    # print("Robot Center of Mass (global coordinates):", robot_com)
    # print(f"force on right foot: {right_foot_force}\nforce on left foot: {left_foot_force}\ntorque on right foot: {right_foot_torque}\ntorque on left foot: {left_foot_torque}\n")

# Launch the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    
    # this lets you see contact forces
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
    # viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

    
    start = time.time()
    new_time = start
    while viewer.is_running() and time.time() - start < 100:
        # check if the simulaiton is paused
        if not paused:
            step_start = time.time()
            
            # Check if we need to switch the leg phase
            time_in_phase += m.opt.timestep
            if time_in_phase >= step_duration:
                left_leg_swing = not left_leg_swing  # Toggle leg phase
                time_in_phase = 0  # Reset phase timer

            # Balance control to keep torso upright
            # balance_control()
            # move_legs()
            stand_up_from_squat()
            test_sensors()
            
            # Step the simulation
            mujoco.mj_step(m, d)
            
        # Check for pause input (e.g., a key press or external trigger)
        # Here we just simulate a pause toggle after 5 seconds for demonstration
        # if time.time() - new_time > 0.01 and not paused:
        #     new_time = time.time()
        #     toggle_pause()
            
        # if time.time() - new_time > 5 and paused:
        #     new_time = time.time()
        #     toggle_pause()
        
        
        # Toggle contact points every two seconds for debugging
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Sync viewer and update
        viewer.sync()

        # Timekeeping to match the simulation timestep
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

# russ t drake - underactuated robotics
# gyroscope, accelerometer, magnetometer, analogue pressure sensor, 