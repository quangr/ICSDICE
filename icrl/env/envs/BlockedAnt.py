import jax.numpy as jnp
import jax
from icrl.common.DataReader import get_dataset
env_id="Ant-v4"

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (jnp.abs(info['true_obs'][:,20])+jnp.abs(info['true_obs'][:,22])+jnp.abs(info['true_obs'][:,24])+jnp.abs(info['true_obs'][:,26])>1)
    else:
        return  (jnp.abs(next_obs[:,20])+jnp.abs(next_obs[:,22])+jnp.abs(next_obs[:,24])+jnp.abs(next_obs[:,26])>1)


def offline_dataset():
    dataset1=get_dataset("dataset/BlockedAnt/suboptimal/",100000,200000)
    dataset2=get_dataset("dataset/BlockedAnt/suboptimal_reward_shaping",100000,200000)
    return jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]),  dataset1,dataset2
        )
