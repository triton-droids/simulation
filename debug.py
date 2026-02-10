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
    "policies/new_locomotion.pt": envs[1],
    "policies/phase_obs_kinda_jumpy.pt": envs[1],
    "policies/phase_obs_contact_pen.pt": envs[1],
    "policies/phase_obs_contact_pen2.pt": envs[1]
}

def _extract_frame_obs(obs: np.ndarray, env, frame_idx: int) -> np.ndarray:
    """Extract one stacked frame from flattened observation using env stack layout."""
    single_frame_size = env._single_frame_size
    frame_stack = env._frame_stack

    if frame_stack <= 1:
        return obs[:single_frame_size]

    # Locomotion env can flatten stack as [features, frames] (feature-major).
    if hasattr(env, "_stack_frame_major"):
        if bool(env._stack_frame_major):
            start_idx = frame_idx * single_frame_size
            return obs[start_idx:start_idx + single_frame_size]
        return obs[frame_idx::frame_stack][:single_frame_size]

    # Legacy layout: contiguous frame chunks.
    start_idx = frame_idx * single_frame_size
    return obs[start_idx:start_idx + single_frame_size]

def _format_named(values: np.ndarray, names: list[str]) -> str:
    pairs = [f"{name}={float(value):.6f}" for name, value in zip(names, values)]
    return ", ".join(pairs)

def format_obs_detailed(obs: np.ndarray, env, include_height:bool) -> str:
    frame_stack = env._frame_stack
    num_dofs = env._nu
    if hasattr(env, "_policy_joint_order") and len(env._policy_joint_order) == num_dofs:
        joint_names = list(env._policy_joint_order)
    else:
        joint_names = [f"joint_{i}" for i in range(num_dofs)]
    lines = []
    for i in range(frame_stack):
        frame_obs = _extract_frame_obs(obs, env, i)
        idx = 0

        # Height is optional for older policies/envs.
        # New locomotion env ignores height in policy observations.
        if include_height:
            height = frame_obs[idx:idx + 1]
            if height.size == 1:
                idx += 1
            else:
                height = None

        lin_vel_cmd = frame_obs[idx:idx+3]
        idx += 3

        ang_vel_cmd_scaled = frame_obs[idx:idx+3]
        idx += 3

        up_cmd = frame_obs[idx:idx+3]
        idx += 3

        commands = frame_obs[idx:idx+3]
        idx += 3

        act_pos_scaled = frame_obs[idx:idx+num_dofs]
        idx += num_dofs

        act_vel_scaled = frame_obs[idx:idx+num_dofs]
        idx += num_dofs

        prev_actions = frame_obs[idx:idx+num_dofs]

        lines.append(f"[ObsDebug] env=0 frame={i}")
        if include_height and height is not None:
            lines.append(f"  height: {height.tolist()}")
        if i == 0 and hasattr(env, "_policy_joint_order"):
            lines.append(f"  joint_order(interleaved): {env._policy_joint_order}")
        lines.append(f"  lin_vel_cmd: {lin_vel_cmd.tolist()}")
        lines.append(f"  ang_vel_cmd_scaled: {ang_vel_cmd_scaled.tolist()}")
        lines.append(f"  up_cmd: {up_cmd.tolist()}")
        lines.append(f"  commands: {commands.tolist()}")
        lines.append(f"  act_pos_scaled: {act_pos_scaled.tolist()}")
        lines.append(f"  act_pos_scaled_named: {_format_named(act_pos_scaled, joint_names)}")
        lines.append(f"  act_vel_scaled: {act_vel_scaled.tolist()}")
        lines.append(f"  act_vel_scaled_named: {_format_named(act_vel_scaled, joint_names)}")
        lines.append(f"  prev_actions: {prev_actions.tolist()}")
        if i < frame_stack - 1:
            lines.append("")
    return "\n".join(lines)
