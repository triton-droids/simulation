""" Domain randomization """

import jax
from brax import base
from src.robots.robot import Robot
from typing import Tuple, List

def domain_randomize(
        sys: base.System, 
        rng: jax.Array,
        robot: Robot,
        friction_range: List[float],
        frictionloss_range: List[float],
        armature_range: List[float],
        body_mass_range: List[float],
        torso_mass_range: List[float],
        qpos0_range: List[float],
        ) -> Tuple[base.System, base.System]:
    """ Randomizes the physical parameters of a system within specified ranges"""
    @jax.vmap
    def rand(rng):
        #Set geom friction
        rng, key = jax.random.split(rng)
        geom_friction = sys.geom_friction.at[:, 0].set(
            jax.random.uniform(key, minval=friction_range[0], maxval=friction_range[1])
        )

        #Scale static friction (excluding free joint)
        rng, key = jax.random.split(rng)
        frictionloss = sys.dof_frictionloss[6:] * jax.random.uniform(
            key, shape=(sys.nu,), minval=frictionloss_range[0], maxval=frictionloss_range[1]
        )
        dof_frictionloss = sys.dof_frictionloss.at[6:].set(frictionloss)
    
        #Scale armature (excluding free joint)
        rng, key = jax.random.split(rng)
        armature = sys.dof_armature[6:] * jax.random.uniform(
            key, shape=(sys.nu,), minval=armature_range[0], maxval=armature_range[1]
        )
        dof_armature = sys.dof_armature.at[6:].set(armature)

        # Scale all link masses: 
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(
            key, shape=(sys.nbody,), minval=body_mass_range[0], maxval=body_mass_range[1]
        )
        body_mass = sys.body_mass.at[:].set(sys.body_mass * dmass)

        # Add mass to torso: 
        rng, key = jax.random.split(rng)
        dmass = jax.random.uniform(key, minval=torso_mass_range[0], maxval=torso_mass_range[1])
        body_mass = body_mass.at[robot.bodies["torso"]["body_id"]].set(
            body_mass[robot.bodies["torso"]["body_id"]] + dmass
        )

        # Jitter qpos0: +U(-0.05, 0.05).
        rng, key = jax.random.split(rng)
        qpos0 = sys.qpos0
        qpos0 = qpos0.at[7:].set(
            qpos0[7:]
            + jax.random.uniform(key, shape=(sys.nu,), minval=qpos0_range[0], maxval=qpos0_range[1])
        )

        return (
            geom_friction,
            dof_frictionloss,
            dof_armature,
            body_mass,
            qpos0,
        )

    (
        friction,
        frictionloss,
        armature,
        body_mass,
        qpos0,
    ) = rand(rng)

    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0,
        "dof_frictionloss": 0,
        "dof_armature": 0,
        "body_mass": 0,
        "qpos0": 0,
    })

    sys = sys.tree_replace({
        "geom_friction": friction,
        "dof_frictionloss": frictionloss,
        "dof_armature": armature,
        "body_mass": body_mass,
        "qpos0": qpos0,
    })

    return sys, in_axes
