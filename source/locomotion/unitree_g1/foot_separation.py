"""Bounded penalty for narrow/crossed feet in the robot heading frame."""
import jax.numpy as jp

def narrow_foot_cost(feet, quaternion, minimum_width=0.16):
    w,x,y,z = quaternion
    yaw = jp.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    delta = feet[0]-feet[1]  # pinned site order: left, right
    width = -jp.sin(yaw)*delta[0]+jp.cos(yaw)*delta[1]
    return jp.square(jp.clip((minimum_width-width)/minimum_width, 0., 1.))
