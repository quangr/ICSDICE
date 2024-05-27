env_id="Hopper-v4"
import jax.numpy as jnp
from flax.training import orbax_utils
import orbax.checkpoint
import jax
from icrl.common.DataReader import get_dataset
def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (jnp.abs(info['true_obs'][:,4])>0.3)
    else:
        return  (jnp.abs(next_obs[:,4])>0.3)

def offline_dataset():
    dataset1=get_dataset("dataset/BlockedHopper/suboptimal/",100000,200000)
    dataset2=get_dataset("dataset/BlockedHopper/suboptimal_reward_shaping/",100000,200000)
    return jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]),  dataset1,dataset2
        )
