import jax.numpy as jnp
import jax
import numpy as np
import gymnasium
import flax
import jax.numpy as jnp
from icrl.common.DataReader import get_dataset

from icrl.common.venv_wrappers import EnvWrapper

env_id = "HalfCheetah-v4"


@flax.struct.dataclass
class RewardWrapper(EnvWrapper):
    def recv(self, ret):
        next_obs, reward, next_done, next_truncated, info = ret
        reward = (info["reward_ctrl"] + jnp.abs(info["reward_run"])).astype(jnp.float32)
        return self, (next_obs, reward, next_done, next_truncated, info)


@flax.struct.dataclass
class AddObsWrapper(EnvWrapper):
    def recv(self, ret):
        next_obs, reward, next_done, next_truncated, info = ret
        return self, (
            jnp.concatenate([next_obs, info["x_position"][:, None]], axis=-1),
            reward,
            next_done,
            next_truncated,
            info,
        )

    def reset(self, ret):
        obs, info = ret
        return self, (
            jnp.concatenate([obs, info["x_position"][:, None]], axis=-1),
            info,
        )


def get_wrappers(envs):
    return [RewardWrapper(), AddObsWrapper()]


def cost_function(next_obs, reward, next_done, next_truncated, info):
    return info["x_position"] < -3


observation_space = gymnasium.spaces.Box(
    float("-inf"), float("inf"), shape=(18,), dtype=np.float64
)


def offline_dataset():
    dataset1 = get_dataset(
        "dataset/HalfCheetahWithObstacle/suboptimal",
        100000,
        200000,
    )
    dataset2 = get_dataset(
        "dataset/HalfCheetahWithObstacle/suboptimal_reward_shaping",
        100000,
        200000,
    )
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), dataset1, dataset2)
