import jax.numpy as jnp
import jax
env_id="Ant-v4"
from icrl.common.DataReader import get_dataset

def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,13]>0.5)
    else:
        return (next_obs[:,13]>0.5)



def offline_dataset():
    dataset1=get_dataset("dataset/Ant_ls/suboptimal",100000,200000)
    dataset2=get_dataset("dataset/Ant_ls/suboptimal_reward_shaping",100000,200000)
    return jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]),  dataset1,dataset2
        )