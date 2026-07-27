"""Collision helper functions for MJX locomotion environments."""

import jax.numpy as jp

def check_feet_contact(pipeline_state, feet_link_ids):
    """Check whether each foot link is touching the ground.

    Args:
        pipeline_state: Current Brax/MJX pipeline state with contact data.
        feet_link_ids: Link IDs corresponding to the feet.

    Returns:
        Boolean JAX array with one entry per foot.
    """

    contact = jp.array([
        jp.any(((pipeline_state.contact.link_idx[0] == -1) & jp.isin(pipeline_state.contact.link_idx[1], jp.array([foot]))) |
               ((pipeline_state.contact.link_idx[1] == -1) & jp.isin(pipeline_state.contact.link_idx[0], jp.array([foot]))))
        for foot in feet_link_ids
    ])
    return contact
