import numpy as np

envs = ["disturbance_env", "locomotion_env"]

# Determine if the policy requires height in observations
height_policies = [
    "policies/disturbance_rejection.pt",
    "policies/locomotion_dr.pt",
    "policies/locomotion.pt"
]

policies = {
    "policies/disturbance_rejection.pt": envs[0],
    "policies/locomotion_dr.pt": envs[1],
    "policies/locomotion.pt": envs[1],
    "policies/locomotion_v2.pt": envs[1],
    "policies/locomotion_v2_dr.pt": envs[1],
}

def format_obs_detailed(obs: np.ndarray, env, include_height:bool) -> str:
    single_frame_size = env._single_frame_size
    frame_stack = env._frame_stack
    num_dofs = env._nu
    lines = []
    for i in range(frame_stack):
        start_idx = i * single_frame_size
        idx = start_idx

        if include_height:
            height = obs[idx:idx+1]
            idx += 1

        lin_vel_cmd = obs[idx:idx+3]
        idx += 3

        ang_vel_cmd_scaled = obs[idx:idx+3]
        idx += 3

        up_cmd = obs[idx:idx+3]
        idx += 3

        commands = obs[idx:idx+3]
        idx += 3

        act_pos_scaled = obs[idx:idx+num_dofs]
        idx += num_dofs

        act_vel_scaled = obs[idx:idx+num_dofs]
        idx += num_dofs

        prev_actions = obs[idx:idx+num_dofs]

        lines.append(f"[ObsDebug] env=0 frame={i}")
        if include_height:
            lines.append(f"  height: {height.tolist()}")
        lines.append(f"  lin_vel_cmd: {lin_vel_cmd.tolist()}")
        lines.append(f"  ang_vel_cmd_scaled: {ang_vel_cmd_scaled.tolist()}")
        lines.append(f"  up_cmd: {up_cmd.tolist()}")
        lines.append(f"  commands: {commands.tolist()}")
        lines.append(f"  act_pos_scaled: {act_pos_scaled.tolist()}")
        lines.append(f"  act_vel_scaled: {act_vel_scaled.tolist()}")
        lines.append(f"  prev_actions: {prev_actions.tolist()}")
        if i < frame_stack - 1:
            lines.append("")
    return "\n".join(lines)