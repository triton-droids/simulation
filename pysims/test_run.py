import time
import numpy as np
import mujoco
import mujoco.viewer
import jax as j
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import model

# Load the humanoid model
m = mujoco.MjModel.from_xml_path('/Users/darin/desktop/_/sims/pysims/models/humanoid.xml')
d = mujoco.MjData(m)

# Load the PPO policy
params_path = "/Users/darin/desktop/_/sims/ksim/ksim/mjx_gym/weights/default_humanoid_walkdefault_run.pkl"
params = model.load_params(params_path)

# Define observation and action sizes based on humanoid.xml
observation_size = 376  # Update based on your humanoid.xml
action_size = 17        # Update based on your humanoid.xml

# Create PPO policy network
policy_network = ppo_networks.make_ppo_networks(
    observation_size,
    action_size,
    preprocess_observations_fn=lambda obs, _: obs,  # Accepts two arguments but ignores the second
    policy_hidden_layer_sizes=[64, 64],  # Update with your PPO network architecture
    value_hidden_layer_sizes=[64, 64],
)

params = (params[0], params[1].policy)
inference_fn = ppo_networks.make_inference_fn(policy_network)(params)

# Initialize JAX random key
rng = j.random.PRNGKey(0)

# Launch the viewer
with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    while viewer.is_running() and time.time() - start < 100:
        step_start = time.time()

        # Gather observations
        observations = np.concatenate((d.qpos, d.qvel))

        # Generate actions using PPO policy (pass the key_sample)
        actions = inference_fn(observations, rng)

        # Update RNG for next step
        rng, _ = j.random.split(rng)

        # Apply actions to the humanoid
        d.ctrl[:] = actions

        # Step the simulation
        mujoco.mj_step(m, d)

        # Debug contact points
        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

        # Sync viewer and update
        viewer.sync()

        # Match simulation timestep
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
