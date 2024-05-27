import jax.numpy as jnp
import jax
from icrl.common.DataReader import get_dataset

from flax.training import orbax_utils
import orbax.checkpoint
env_id="Walker2d-v4"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,8]>1)
    else:
        return (next_obs[:,8]>1)


def offline_dataset():
    dataset1 = get_dataset(
        "dataset/Walker_ls/suboptimal",
        100000,
        200000,
    )
    dataset2 = get_dataset(
        "dataset/Walker_ls/suboptimal_reward_shaping",
        100000,
        200000,
    )
    return jax.tree.map(lambda x, y: jnp.concatenate([x, y]), dataset1, dataset2)
