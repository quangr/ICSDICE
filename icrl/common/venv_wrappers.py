import envpool

import flax
import jax.numpy as jnp

from .RunningMeanStd import RunningMeanStd


class VectorEnvWrapper:
    def __init__(self, env, wrappers):
        self.env = env
        self.wrappers = wrappers
        self.handles = []
        self._handle, self._recv, self._send, self._step = self.env.xla()

    @property
    def action_space(self):
        """Returns the action space of the environment."""
        return self.env.action_space

    @property
    def single_action_space(self):
        """Returns the action space of the environment."""
        return self.env.action_space

    @property
    def single_observation_space(self):
        return self.observation_space

    @property
    def observation_space(self):
        """Returns the action space of the environment."""
        if hasattr(self, "_observation_space"):
            return self._observation_space
        else:
            return self.env.observation_space

    def set_observation_space(self, observation_space):
        """Returns the action space of the environment."""
        self._observation_space = observation_space

    def reset(self):
        result = self.env.reset()
        handles = [self._handle]
        for wrapper in reversed(self.wrappers):
            handle, result = wrapper.reset(result)
            handles += [handle]
        self.handles = handles[1:]
        return handles, result

    def step(self, action):
        for wrapper in self.wrappers:
            action = wrapper.send(action)
        results = self.env.step(action)
        for handle in self.handles:
            _, results = handle.recv(results)
        return results

    def xla(self):
        def _apply_handle(ret, x):
            f, handle = x
            newhandle, ret = f(handle, ret)
            return ret, newhandle

        def recv(handles: jnp.ndarray):
            _handle, ret = self._recv(handles[0])
            new_handles = []
            # reversed
            for handle in handles[1:]:
                handle, ret = handle.recv(ret)
                new_handles += [handle]
            return [_handle] + new_handles, ret

        def send(handle: jnp.ndarray, action, env_id=None):
            for wrapper in self.wrappers:
                action = wrapper.send(action)
            return [self._send(handle[0], action, env_id)] + handle[1:]

        def step(handle, action, env_id=None):
            return recv(send(handle, action, env_id))

        return self._handle, recv, send, step


@flax.struct.dataclass
class EnvWrapper:
    def recv(self, ret):
        return self, ret

    def reset(self, ret):
        return self, ret

    def send(self, action):
        return action


@flax.struct.dataclass
class VectorEnvNormObs(EnvWrapper):
    obs_rms: RunningMeanStd = RunningMeanStd()

    def recv(self, ret):
        _next_obs, reward, next_done, next_truncated, info = ret
        obs_rms = self.obs_rms.update(_next_obs)
        return self.replace(obs_rms=obs_rms), (
            obs_rms.norm(_next_obs),
            reward,
            next_done,
            next_truncated,
            info | {"true_obs": _next_obs},
        )

    def reset(self, ret):
        _obs, info = ret
        obs_rms = self.obs_rms.update(_obs)
        obs = obs_rms.norm(_obs).astype(jnp.float32)
        return self.replace(obs_rms=obs_rms), (obs, info | {"true_obs": _obs})


@flax.struct.dataclass
class MojocoEnvDtypeAct(EnvWrapper):
    def send(self, action):
        return action.astype(jnp.float64)


@flax.struct.dataclass
class VectorEnvClipAct(EnvWrapper):
    action_low: jnp.array
    action_high: jnp.array

    def send(self, action):
        action_remap = jnp.clip(action, -1.0, 1.0)
        action_remap = (
            self.action_low
            + (action_remap + 1.0) * (self.action_high - self.action_low) / 2.0
        )
        return action_remap


if __name__ == "__main__":
    envs = envpool.make(
        "HalfCheetah-v3",
        env_type="gym",
        num_envs=2,
        seed=0,
    )
    wrappers = [
        VectorEnvNormObs(),
        VectorEnvClipAct(envs.action_space.low, envs.action_space.high),
    ]
    a = VectorEnvWrapper(envs, wrappers)
    handle, recv, send, step_env = a.xla()
    handle, s = a.reset()
    send(handle, jnp.array([[0.0] * 6] * 2))
    # print(jax.make_jaxpr(send)(handle,jnp.array([[0.]*6]*2)))
    recv(handle)
    print(step_env(handle, jnp.array([[0.0] * 6] * 2)))
