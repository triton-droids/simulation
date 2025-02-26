import mujoco
import numpy as np
from mujoco import viewer
import time

def get_sensor_values(data):
    """
    Read sensor values from the humanoid environment.
    
    Args:
        env: MuJoCo environment with humanoid model
    
    Returns:
        dict: Dictionary containing all sensor readings
    """
   
    
    sensor_data = {
        # IMU sensors
        'gyro': data.sensor('gyro').data.copy(),
        'local_linvel': data.sensor('local_linvel').data.copy(),
        'accelerometer': data.sensor('accelerometer').data.copy(),
        'upvector': data.sensor('upvector').data.copy(),
        'forwardvector': data.sensor('forwardvector').data.copy(),
        'global_linvel': data.sensor('global_linvel').data.copy(),
        'global_angvel': data.sensor('global_angvel').data.copy(),
        'position': data.sensor('position').data.copy(),
        'orientation': data.sensor('orientation').data.copy(),
        
        # Foot sensors
        'l_foot_global_linvel': data.sensor('l_foot_global_linvel').data.copy(),
        'r_foot_global_linvel': data.sensor('r_foot_global_linvel').data.copy(),
        'l_foot_upvector': data.sensor('l_foot_upvector').data.copy(),
        'r_foot_upvector': data.sensor('r_foot_upvector').data.copy(),
        'l_foot_pos': data.sensor('l_foot_pos').data.copy(),
        'r_foot_pos': data.sensor('r_foot_pos').data.copy(),
    }
    
    return sensor_data

def print_sensor_values(sensor_data):
    """Print all sensor values in a formatted way."""
    print("\n=== Humanoid Sensor Readings ===")
    
    print("\nIMU Readings:")
    print(f"Gyroscope: {sensor_data['gyro']}")
    print(f"Local Linear Velocity: {sensor_data['local_linvel']}")
    print(f"Accelerometer: {sensor_data['accelerometer']}")
    print(f"Up Vector: {sensor_data['upvector']}")
    print(f"Forward Vector: {sensor_data['forwardvector']}")
    print(f"Global Linear Velocity: {sensor_data['global_linvel']}")
    print(f"Global Angular Velocity: {sensor_data['global_angvel']}")
    print(f"Position: {sensor_data['position']}")
    print(f"Orientation (quaternion): {sensor_data['orientation']}")
    
    print("\nFoot Readings:")
    print(f"Left Foot Global Linear Velocity: {sensor_data['l_foot_global_linvel']}")
    print(f"Right Foot Global Linear Velocity: {sensor_data['r_foot_global_linvel']}")
    print(f"Left Foot Up Vector: {sensor_data['l_foot_upvector']}")
    print(f"Right Foot Up Vector: {sensor_data['r_foot_upvector']}")
    print(f"Left Foot Position: {sensor_data['l_foot_pos']}")
    print(f"Right Foot Position: {sensor_data['r_foot_pos']}")

# Example usage:
if __name__ == "__main__":
    # Initialize MuJoCo and load the model
    m = mujoco.MjModel.from_xml_path("text.xml")
    d = mujoco.MjData(m)
    
    # Set initial height (e.g., 10 meters above ground)
    d.qpos[2] = 10.0  # The third element of qpos controls the vertical position
    mujoco.mj_forward(m, d)  # Update all derived quantities
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < 30:
            step_start = time.time()

            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)
            sensor_data = get_sensor_values(d)
       
        
        # Print the readings (optional - comment out if too verbose)
            print_sensor_values(sensor_data)

           

            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

  