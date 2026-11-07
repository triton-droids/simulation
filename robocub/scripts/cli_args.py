"""Shared command-line arguments for training and checkpoint workflows."""

import argparse

def add_rl_args(parser: argparse.ArgumentParser):
    """Add RL arguments to the parser.

    Args:
        parser: The parser to add the arguments to.
    """
    # create a new argument group
    arg_group = parser.add_argument_group("brax", description="Arguments for Brax agent.")
    # -- load arguments
    arg_group.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help=(
            "Warm-start normalizer/policy/value parameters from a checkpoint; "
            "optimizer, step, and PRNG state are not restored."
        ),
    )
    arg_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory used for a parameter warm start.",
    )
  
    arg_group.add_argument(
        "--log_project_name", type=str, default=None, help="Name of the logging project when using wandb or neptune."
    )
