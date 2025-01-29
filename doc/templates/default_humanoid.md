Here is the YML template for default humanoid locomotion task.  

```
seed: 

env:
  name: 
  num_joints: 12
  ctrl_dt: 
  sim_dt: 
  episode_length: 
  action_repeat: 
  action_scale: 
  history_len: 
  soft_joint_pos_limit_factor: 
  noise_config:
    level: 
    scales:
      hip_pos: 
      kfe_pos: 
      ffe_pos: 
      faa_pos:
      joint_vel: 
      gravity: 
      linvel: 
      gyro:
    
  d_randomization_config: 
    enable: 
    dynamics: 
      floor_friction:
        op: 
        range: 
      static friction: 
        op: 
        range: 
      armature:
        op: 
        range: 
      link_mass:
        op: 
        range: 
      torso_mass:
        op: 
        range:
      jitter_qpos0:
        op: "
        range: 

  
  reward_config:
    scales: 
      tracking_lin_vel: 
      tracking_ang_vel: 
      lin_vel_z: 
      ang_vel_xy: 
      orientation: 
      base_height: 
      torques:
      action_rate: 
      energy: 
      feet_clearance:
      feet_air_time: 
      feet_slip: 
      feet_height: 
      feet_phase: 
      stand_still:
      alive: 
      termination:
      joint_deviation_knee:
      joint_deviation_hip: 
      dof_pos_limits:
      pose:
    
    tracking_sigma: 
    max_foot_height: 
    base_height_target:
  
  push_config:
    enable:
    interval_range:
    magnitude_range: 
  
  lin_vel_x: 
  lin_vel_y: 
  ang_vel_yaw: 

logger:
  tensorboard: 
  wandb: 
  project_name: 

brax_ppo_agent:
  num_timesteps: 
  num_evals: 
  episode_length: 
  normalize_observations: 
  action_repeat: 
  unroll_length:
  num_minibatches: 
  num_updates_per_batch:
  discounting: 
  learning_rate: 
  entropy_cost: 
  num_envs: 
  batch_size: 
  max_grad_norm: 
  network_factory:
    policy_hidden_layer_sizes: 
    value_hidden_layer_sizes: 
    policy_obs_key: 
    value_obs_key: 
  num_resets_per_eval : 
  
  ```