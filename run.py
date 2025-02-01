import argparse
from mjx.scripts import brax_train_policy, run_policy, evaluate_policy
from mjx.utils.parse import parse_cfg
from mjx.utils.registry import ALL_ENVS
from mjx.utils import registry
from mjx.envs.wrappers import RecordVideo


def main():
    parser = argparse.ArgumentParser(description='Humanoid Robot RL Simulator')
    subparsers = parser.add_subparsers(dest="command", required=True)

     # Training Subparser
    train_parser = subparsers.add_parser("train", help="Train a policy")
    train_parser.add_argument("--name", type=str, required=False, 
                             help="Name of experiment")
    train_parser.add_argument("--env", type=str, 
                              choices=["DefaultHumanoidJoystickFlatTerrain",
                                       "DefaultHumanoidJoystickRoughTerrain"], 
                              required=True, 
                             help="Environment ID to train on") ,
    train_parser.add_argument("--framework", choices=["brax_ppo"],
                            required=True, help="RL training framework")  
    train_parser.add_argument("--checkpoint", type=str, 
                             help="Path to model checkpoint to resume training")
    train_parser.add_argument("--record", type=int, default=1000,
                             help="Record videos during training")

    # Play Subparser
    play_parser = subparsers.add_parser("play", help="Run a trained policy")
    play_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to trained model (e.g., trained_models/model.pth)")
    play_parser.add_argument("--env", type=str,
                            help="Environment ID to run the policy")
    play_parser.add_argument("--render", action="store_true",
                            help="Render simulation in GUI")

    args = parser.parse_args()
    default_cfg = registry.get_default_config(args.env)

    #Merge custom configurations with default configurations
    cfg = parse_cfg('instruct.yml', default_cfg_path=default_cfg)

    env = registry.load(args.env, cfg.env)

    print(f"Environment Config:\n{cfg.env}")

    # #Wrap environment with video recording wrapper to record videos during training
    # if args.record:
    #     env = RecordVideo(env, video_folder="videos", episode_trigger=None, step_trigger=None, video_length=0, name_prefix="rl-video") 
    #NOT FINISHED - Need to store global number of episodes to trigger recording 

    if args.command == "train":
        if args.framework == "brax_ppo":
            print(f"PPO Training Parameters:\n{cfg.brax_ppo_agent}")
            brax_train_policy(
                env,
                cfg, 
                env_name = args.env,
                exp_name=args.name,
                checkpoint=args.checkpoint
                )
        else:
            raise NotImplementedError(f"Framework {args.framework} is not implemented")
        
    elif args.command == "play":
        run_policy(checkpoint_path=args.checkpoint, env_id=args.env, render=args.render)

    elif args.command == "evaluate":
        evaluate_policy(checkpoint_path=args.checkpoint, env_id=args.env)
 
if __name__ == "__main__":
    main()