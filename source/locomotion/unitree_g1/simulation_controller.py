"""Simulation-only command interface for the saved T02 development policy."""
from pathlib import Path
import math

DEFAULT_RUN = Path('results/post_f1_t02/rate_cost/train')
CHECKPOINT = 1003520


def validate_command(command):
    values = tuple(float(x) for x in command)
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise ValueError('Command must be three finite values: vx, vy, yaw_rate')
    if not (-.25 <= values[0] <= .45 and abs(values[1]) <= .25 and abs(values[2]) <= .4):
        raise ValueError('Prototype envelope: vx[-.25,.45] m/s, vy[-.25,.25] m/s, yaw[-.4,.4] rad/s; combinations are not all validated')
    return values


class SimulationController:
    """Owns simulated robot state. No hardware transport or automatic fall reset."""
    def __init__(self, run_dir=DEFAULT_RUN):
        import functools
        import jax
        import jax.numpy as jp
        from brax.io import model
        from brax.training.agents.ppo import networks
        from omegaconf import OmegaConf
        from source.scripts import evaluate_g1 as ev
        self.run_dir = Path(run_dir).resolve()
        policy = self.run_dir/'logs/checkpoints'/str(CHECKPOINT)/'policy'
        if not policy.is_file() or not (self.run_dir/'resolved_config.json').is_file():
            raise FileNotFoundError(f'T02 checkpoint/config missing in {self.run_dir}; provision the handoff artifacts first')
        cfg = OmegaConf.load(self.run_dir/'resolved_config.json')
        cfg.robot.fetch_model = False
        cfg.sim.playground.fetch_source = False
        cfg.sim.reset.randomize = False
        cfg.sim.noise.add_noise = False
        cfg.sim.push.add_push = False
        cfg.sim.domain_rand.add_domain_rand = False
        cfg.sim.playground.recovery_reset_candidates = 1
        cfg.sim.playground.reset_disturbance_scale = 1.
        self.env = ev.get_env_class(cfg.env.name)(cfg.robot.name, ev.make_robot(cfg.robot), cfg.env.terrain, cfg.sim)
        factory = functools.partial(networks.make_ppo_networks,
            policy_hidden_layer_sizes=cfg.agent.policy_hidden_layer_sizes,
            value_hidden_layer_sizes=cfg.agent.value_hidden_layer_sizes,
            policy_obs_key='state', value_obs_key='privileged_state')
        net = ev._make_evaluation_network(factory,self.env.observation_size,self.env.action_size,
            normalize_observations=bool(cfg.agent.normalize_observations))
        self.params = model.load_params(policy)
        infer = networks.make_inference_fn(net)(self.params, deterministic=True)
        def tick(state, command, key):
            state = ev._replace_command(state, command)
            action, _ = infer(state.obs, key)
            return ev._step_with_held_command(self.env,state,action,command)
        self._tick = jax.jit(tick)
        self._reset = jax.jit(self.env.reset)
        self.dt = float(self.env.dt)
        self.state = None
        self.command = (0.,0.,0.)

    def reset(self, seed=6000):
        import jax
        from source.scripts.evaluate_g1 import _replace_command
        import jax.numpy as jp
        reset_key, self._key = jax.random.split(jax.random.PRNGKey(seed))
        self.command = (0.,0.,0.)
        self.state = _replace_command(self._reset(reset_key),jp.zeros(3))
        return self.state

    def set_command(self, vx, vy, yaw_rate):
        self.command = validate_command((vx,vy,yaw_rate))

    def stop(self):
        """Request standing; does not freeze physics or guarantee immediate stopping."""
        self.set_command(0.,0.,0.)

    def step(self):
        import jax
        import jax.numpy as jp
        if self.state is None:
            raise RuntimeError('Call reset() before step()')
        if bool(self.state.done):
            raise RuntimeError('Episode terminated; inspect failure before explicit reset()')
        self._key, action_key = jax.random.split(self._key)
        self.state = self._tick(self.state,jp.asarray(self.command),action_key)
        return self.state

