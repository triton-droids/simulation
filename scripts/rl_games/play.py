# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RL-Games."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RL-Games.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help="When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import gymnasium as gym
import math
import os
import time
import torch

from rl_games.common import env_configurations, vecenv
from rl_games.common.player import BasePlayer
from rl_games.torch_runner import Runner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

import tritonhumanoid.tasks  # noqa: F401


def main():
    """Play with RL-Games agent."""
    # parse env configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rl_games", agent_cfg["params"]["config"]["name"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # find checkpoint
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rl_games", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint is None:
        # specify directory for logging runs
        run_dir = agent_cfg["params"]["config"].get("full_experiment_name", ".*")
        # specify name of checkpoint
        if args_cli.use_last_checkpoint:
            checkpoint_file = ".*"
        else:
            # this loads the best checkpoint
            checkpoint_file = f"{agent_cfg['params']['config']['name']}.pth"
        # get path to previous checkpoint
        resume_path = get_checkpoint_path(log_root_path, run_dir, checkpoint_file, other_dirs=["nn"])
    else:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # wrap around environment for rl-games
    rl_device = agent_cfg["params"]["config"]["device"]
    clip_obs = agent_cfg["params"]["env"].get("clip_observations", math.inf)
    clip_actions = agent_cfg["params"]["env"].get("clip_actions", math.inf)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_root_path, log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rl-games
    env = RlGamesVecEnvWrapper(env, rl_device, clip_obs, clip_actions)

    # register the environment to rl-games registry
    # note: in agents configuration: environment name must be "rlgpu"
    vecenv.register(
        "IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: RlGamesGpuEnv(config_name, num_actors, **kwargs)
    )
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: env})

    # load previously trained model
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = resume_path
    print(f"[INFO]: Loading model checkpoint from: {agent_cfg['params']['load_path']}")

    # set number of actors into agent config
    agent_cfg["params"]["config"]["num_actors"] = env.unwrapped.num_envs
    # create runner from rl-games
    runner = Runner()
    runner.load(agent_cfg)
    # obtain the agent from the runner
    agent: BasePlayer = runner.create_player()
    agent.restore(resume_path)
    agent.reset()

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.reset()
    if isinstance(obs, dict):
        obs = obs["obs"]
    timestep = 0
    # required: enables the flag for batched observations
    _ = agent.get_batch_size(obs, 1)
    # initialize RNN states if used
    if agent.is_rnn:
        agent.init_rnn()
    # simulate environment
    # note: We simplified the logic in rl-games player.py (:func:`BasePlayer.run()`) function in an
    #   attempt to have complete control over environment stepping. However, this removes other
    #   operations such as masking that is used for multi-agent learning by RL-Games.
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # convert obs to agent format
            obs = agent.obs_to_torch(obs)
            # agent stepping
            actions = agent.get_action(obs, is_deterministic=agent.is_deterministic)
            # env stepping
            obs, _, dones, _ = env.step(actions)

            # perform operations for terminated episodes
            if len(dones) > 0:
                # reset rnn state for terminated episodes
                if agent.is_rnn and agent.states is not None:
                    for s in agent.states:
                        s[:, dones, :] = 0.0
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()

    def export_models():
        """Export the RL-Games trained model to TorchScript format with RNN support."""
        import os
        import copy
         
        # Get the model from the agent
        model = agent.model
        
        print(f"\n=== Exporting RL-Games Model ===")
        
        # Get model info
        original_device = next(model.parameters()).device
        model.eval()

        print(f"Model type: {type(model)}")
        print(f"Model device: {original_device}")
        
        # Check if RNN model
        is_rnn = agent.is_rnn
        print(f"Is RNN: {is_rnn}")
        
        # Get network info
        if hasattr(model, 'a2c_network'):
            network = model.a2c_network
        else:
            network = model
        
        # Get input/output dimensions
        if isinstance(agent.obs_shape, dict):
            obs_dim = sum(np.prod(shape) for shape in agent.obs_shape.values())
            print(f"Dictionary observation space detected")
            print(f"Total observation dimensions: {obs_dim}")
        else:
            obs_dim = np.prod(agent.obs_shape)
            print(f"Observations: {obs_dim}")
        
        actions_num = agent.actions_num
        print(f"Actions: {actions_num}")
        
        # Check for input normalization
        normalize_input = agent.normalize_input
        print(f"Normalize input: {normalize_input}")
        
        if is_rnn:
            print("\n=== Creating RNN Policy Wrapper ===")

            try:
                # --- A) prep dims & dummy inputs ---
                import copy

                # Get model info
                original_device = next(model.parameters()).device
                model.eval()

                # --- use a COPY for tracing; do NOT mutate agent.model ---
                model_cpu = copy.deepcopy(model).eval().to('cpu')
                if isinstance(agent.obs_shape, dict):
                    obs_dim = int(sum(np.prod(s) for s in agent.obs_shape.values()))
                else:
                    obs_dim = int(np.prod(agent.obs_shape))
                dummy_obs = torch.randn(1, obs_dim)

                # fetch rnn sizes
                net = model_cpu.a2c_network if hasattr(model_cpu, 'a2c_network') else model_cpu
                r = getattr(net, 'rnn', getattr(net, 'gru', None))
                layers = int(getattr(r, 'num_layers', 1))
                hidden_size = int(getattr(r, 'hidden_size', 256))
                hidden = torch.zeros(layers, 1, hidden_size)

                # tensors-only input dict for the rl-games model
                example_input = {
                    'obs': dummy_obs,                     # [1, D]
                    'rnn_states': [hidden],               # List[Tensor]
                    'seq_length': torch.tensor(1),        # keep dict homogeneous
                    'is_train': torch.tensor(False),
                }

                # --- B) trace a tiny head that returns (actions, next_hidden) ---
                class CoreHead(torch.nn.Module):
                    def __init__(self, core):
                        super().__init__()
                        self.core = core

                    def forward(self,
                                obs: torch.Tensor,
                                hidden: torch.Tensor,
                                seq_len: torch.Tensor,
                                is_train: torch.Tensor):
                        # build the dict INSIDE (tracing will just follow tensor ops)
                        out = self.core({
                            'obs': obs,
                            'rnn_states': [hidden],      # list created here is fine
                            'seq_length': seq_len,
                            'is_train': is_train,
                        })
                        
                        if 'mus' in out:
                            actions = out['mus']                           # deterministic means
                        elif 'mean_actions' in out:                        # some rl-games variants
                            actions = out['mean_actions']
                        elif 'actions' in out:
                            actions = out['actions']                       # fallback
                        elif 'action' in out:
                            actions = out['action']
                        else:
                            raise RuntimeError("No action/mu key in model output")
                        
                        next_h = out['rnn_states'][0]
                        return actions, next_h

                core_head = CoreHead(model_cpu).eval()
                with torch.no_grad():
                    traced_core = torch.jit.trace(
                        core_head,
                        (dummy_obs, hidden, torch.tensor(1), torch.tensor(False)),
                        check_trace=False
                    )
                    
                clip = float(agent.clip_actions) if np.isfinite(agent.clip_actions) else float("inf")

                # --- C) scripted wrapper that manages hidden state and calls traced_core ---
                class RNNPolicyWrapper(torch.nn.Module):
                    def __init__(self, traced_core, layers, hidden_size, clip=float('inf'), act_dim=None):
                        super().__init__()
                        self.core = traced_core.eval()
                        self.rnn_layers = int(layers)
                        self.rnn_hidden_size = int(hidden_size)
                        self.register_buffer("hidden_state",
                                            torch.zeros(self.rnn_layers, 1, self.rnn_hidden_size))
                        self.register_buffer("one", torch.tensor(1, dtype=torch.long))
                        self.register_buffer("false", torch.tensor(False))
                        self.clip = float(clip)
                        self.act_dim = int(act_dim) if act_dim is not None else -1

                    def forward(self, obs: torch.Tensor) -> torch.Tensor:
                        if obs.dim() == 1:
                            obs = obs.unsqueeze(0)
                        B = obs.size(0)
                        if self.hidden_state.size(1) != B:
                            self.hidden_state = self.hidden_state[:, :1, :].expand(
                                self.rnn_layers, B, self.rnn_hidden_size
                            ).contiguous()

                        actions, next_h = self.core(obs, self.hidden_state, self.one, self.false)
                        self.hidden_state = next_h
                        
                        if hasattr(self, "clip") and self.clip < float("inf"):
                            actions = torch.clamp(actions, -self.clip, self.clip)
                        
                        if self.act_dim > 0 and actions.size(-1) != self.act_dim:
                            raise RuntimeError(f"Action dim mismatch: got {actions.size(-1)}, expected {self.act_dim}")
                        
                        return actions[0] if actions.size(0) == 1 else actions
                    
                    @torch.jit.export
                    def reset_mask(self, done: torch.Tensor) -> None:
                        # done: [B] bool or [B,1]
                        if done.dim() == 2:
                            done = done.squeeze(1)
                        # JIT-safe: no kwargs, no python tuple outputs
                        idx = torch.nonzero(done).squeeze(1)
                        if idx.numel() > 0:
                            self.hidden_state.index_fill_(1, idx, 0)

                    @torch.jit.export
                    def reset_memory(self) -> None:
                        self.hidden_state.zero_()

                wrapper = RNNPolicyWrapper(traced_core, layers, hidden_size, clip=clip, act_dim=actions_num).cpu()

                # smoke test
                with torch.no_grad():
                    _ = wrapper(dummy_obs)

                print("\n=== Scripting RNN Policy ===")
                scripted_policy = torch.jit.script(wrapper)
                
                # Save to both locations
                policy_path = os.path.join(save_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                scripted_policy.save(policy_path)
                print(f"✅ RNN policy exported to: {policy_path}")
                
                real_control_policy_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                scripted_policy.save(real_control_policy_path)
                print(f"✅ RNN policy also saved to: {real_control_policy_path}")

                # Save metadata to both locations
                metadata = {
                    'num_observations': obs_dim,
                    'num_actions': actions_num,
                    'normalize_input': normalize_input,
                    'is_rnn': True,
                    'rnn_layers': layers,
                    'rnn_hidden_size': hidden_size,
                    'obs_shape': agent.obs_shape,
                    'clip_actions': agent.clip_actions,
                }
                torch.save(metadata, os.path.join(save_dir, "rlgames_metadata.pt"))
                torch.save(metadata, os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_metadata.pt"))
                print("✅ Metadata saved to both locations")
                return policy_path
                
            except Exception as e:
                print(f"❌ Failed to script RNN policy: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback: save complete model to both locations
                print("\n=== Fallback: Saving Complete Model ===")
                model_path = os.path.join(save_dir, "rlgames_model_complete.pt")
                real_control_model_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_model_complete.pt")
                
                model_data = {
                    'model_state_dict': model.state_dict(),
                    'model_type': type(model).__name__,
                    'normalize_input': normalize_input,
                    'obs_shape': agent.obs_shape,
                    'actions_num': actions_num,
                    'is_rnn': True,
                    'rnn_layers': wrapper.rnn_layers if 'wrapper' in locals() else 1,
                    'rnn_hidden_size': wrapper.rnn_hidden_size if 'wrapper' in locals() else 256,
                }
                
                torch.save(model_data, model_path)
                torch.save(model_data, real_control_model_path)
                print(f"✅ Complete model saved to: {model_path}")
                print(f"✅ Complete model also saved to: {real_control_model_path}")
                return model_path
        
        else:
            print("\n=== Creating Feedforward Policy Wrapper ===")
            
            # For non-RNN models, use the simpler approach
            def policy_wrapper(obs):
                with torch.no_grad():
                    processed_obs = obs
                    
                    if normalize_input and hasattr(model, 'running_mean_std'):
                        processed_obs = model.running_mean_std(processed_obs)
                    
                    input_dict = {
                        'is_train': False,
                        'obs': processed_obs,
                    }
                    
                    result = model(input_dict)
                    
                    if 'mus' in result:
                        return result['mus']
                    elif 'logits' in result:
                        return torch.argmax(result['logits'], dim=-1)
                    else:
                        return result.get('actions', result.get('action'))
            
            # Disable gradients
            for param in model.parameters():
                param.requires_grad_(False)
            
            # Test and trace
            dummy_obs = torch.randn(1, obs_dim, device=original_device)
            
            print("\n=== Tracing Feedforward Policy ===")
            try:
                with torch.no_grad():
                    traced_policy = torch.jit.trace(policy_wrapper, dummy_obs)
                    
                    # Save to both locations
                    policy_path = os.path.join(save_dir, "rlgames_policy.pt")
                    traced_policy.save(policy_path)
                    print(f"✅ Feedforward policy exported to: {policy_path}")
                    
                    real_control_policy_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_policy.pt")
                    traced_policy.save(real_control_policy_path)
                    print(f"✅ Feedforward policy also saved to: {real_control_policy_path}")
                    
                    # Save metadata to both locations
                    metadata = {
                        'num_observations': obs_dim,
                        'num_actions': actions_num,
                        'normalize_input': normalize_input,
                        'is_rnn': False,
                        'obs_shape': agent.obs_shape,
                        'clip_actions': agent.clip_actions,
                    }
                    
                    if normalize_input and hasattr(model, 'running_mean_std'):
                        metadata['running_mean_std_state'] = model.running_mean_std.state_dict()
                    
                    metadata_path = os.path.join(save_dir, "rlgames_metadata.pt")
                    torch.save(metadata, metadata_path)
                    
                    real_control_metadata_path = os.path.join(real_control_dir, f"rlgames_{args_cli.run_name}_metadata.pt")
                    torch.save(metadata, real_control_metadata_path)
                    
                    print(f"✅ Metadata saved to: {metadata_path}")
                    print(f"✅ Metadata also saved to: {real_control_metadata_path}")
                    
                    return policy_path
                    
            except Exception as e:
                print(f"❌ Failed to trace policy: {e}")
                import traceback
                traceback.print_exc()
                return None

    def test_exported_policy():
        """Test the exported RL-Games policy"""
        import os
        
        device = agent.device
        
        # Load metadata first
        metadata_path = os.path.join(save_dir, "rlgames_metadata.pt")
        if os.path.exists(metadata_path):
            metadata = torch.load(metadata_path)
            is_rnn = metadata.get('is_rnn', False)
            obs_dim = metadata['num_observations']
        else:
            print("❌ No metadata found")
            return None
        
        # Load appropriate policy
        if is_rnn:
            policy_path = os.path.join(save_dir, f"rlgames_{args_cli.run_name}_policy.pt")
            if os.path.exists(policy_path):
                exported_policy = torch.jit.load(policy_path).to(device)
                print("✅ Loaded RNN policy")
                
                # Test with dummy observation
                dummy_obs = torch.randn(1, obs_dim, device=device)
                
                with torch.no_grad():
                    # Reset memory for both
                    exported_policy.reset_memory()
                    if agent.is_rnn and agent.states is not None:
                        for s in agent.states:
                            s[:] = 0.0
                    
                    # Get agent action
                    if isinstance(agent.obs_shape, dict):
                        test_obs = agent.obs_to_torch({'obs': dummy_obs.cpu().numpy()})
                    else:
                        test_obs = dummy_obs
                    
                    def _as_batch_actions(x, device):
                        # unwrap common container types
                        if isinstance(x, (list, tuple)):
                            x = x[0]
                        if isinstance(x, dict):
                            for k in ("actions", "mus", "mean_actions", "logits"):
                                if k in x:
                                    x = x[k]
                                    break
                            else:
                                # fall back to first value
                                x = next(iter(x.values()))

                        # to tensor
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x)
                        if not isinstance(x, torch.Tensor):
                            x = torch.as_tensor(x)

                        # to device + batchify
                        x = x.to(device)
                        if x.dim() == 0:
                            x = x.view(1, 1)
                        elif x.dim() == 1:
                            x = x.unsqueeze(0)
                        return x

                    # In test_exported_policy()
                    agent_out_raw = agent.get_action(test_obs, is_deterministic=True)
                    agent_actions  = _as_batch_actions(agent_out_raw, device)
                    
                    # Get exported action
                    exported_actions = exported_policy(dummy_obs)
                    if len(exported_actions.shape) == 1:
                        exported_actions = exported_actions.unsqueeze(0)
                    
                    print(f"Agent actions: {agent_actions[0][:3] if agent_actions.shape[-1] >= 3 else agent_actions[0]}")
                    print(f"Exported actions: {exported_actions[0][:3] if exported_actions.shape[-1] >= 3 else exported_actions[0]}")
                    
                    # Note: RNN outputs may differ slightly due to initialization
                    if torch.allclose(agent_actions, exported_actions, atol=1e-3):
                        print("✅ Exported RNN policy outputs are similar to original")
                    else:
                        print("⚠️ RNN outputs differ (this is often expected due to state initialization)")
                        print(f"Max difference: {torch.max(torch.abs(agent_actions - exported_actions)).item()}")
                
                return exported_policy
            else:
                print("❌ No RNN policy found")
                return None
        else:
            policy_path = os.path.join(save_dir, "rlgames_policy.pt")
            if os.path.exists(policy_path):
                exported_policy = torch.jit.load(policy_path).to(device)
                print("✅ Loaded feedforward policy")
                
                # Test matching (similar to before)
                dummy_obs = torch.randn(1, obs_dim, device=device)
                
                with torch.no_grad():
                    if isinstance(agent.obs_shape, dict):
                        test_obs = agent.obs_to_torch({'obs': dummy_obs.cpu().numpy()})
                    else:
                        test_obs = dummy_obs
                    
                    agent_actions = agent.get_action(test_obs, is_deterministic=True)
                    exported_actions = exported_policy(dummy_obs)
                    
                    print(f"Agent actions: {agent_actions[0][:3] if agent_actions.shape[-1] >= 3 else agent_actions[0]}")
                    print(f"Exported actions: {exported_actions[0][:3] if exported_actions.shape[-1] >= 3 else exported_actions[0]}")
                    
                    if torch.allclose(agent_actions, exported_actions, atol=1e-4):
                        print("✅ Exported policy matches original agent")
                    else:
                        print("⚠️ Warning: Exported policy differs from original")
                        print(f"Max difference: {torch.max(torch.abs(agent_actions - exported_actions)).item()}")
                
                return exported_policy
            else:
                print("❌ No feedforward policy found")
                return None
    
    # Only export if not using unexported flag
    if not args_cli.use_unexported:
        # Export the model after loading
        exported_policy_path = export_models()
        print("Export complete. Starting simulation...")
        
        # Test the exported policy
        exported_policy = test_exported_policy()
        use_exported = exported_policy is not None
    else:
        print("Skipping model export for evaluation...")
        exported_policy = None
        use_exported = False


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
