import jax.numpy as jp

def check_feet_contact(pipeline_state, feet_link_ids):
    contact = jp.array([
        jp.any(((pipeline_state.contact.link_idx[0] == -1) & jp.isin(pipeline_state.contact.link_idx[1], jp.array([foot]))) |
               ((pipeline_state.contact.link_idx[1] == -1) & jp.isin(pipeline_state.contact.link_idx[0], jp.array([foot]))))
        for foot in feet_link_ids
    ])
    return contact