env_id="commonroad"
from icrl.common.DataReader import get_dataset
import numpy as np
import jax.numpy as jnp
import orbax.checkpoint
import jax
def cost_function(next_obs, reward, next_done, next_truncated, info):
        return (next_obs[:,22]<20) | (next_obs[:,22]>100)



def offline_dataset():
        checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        expert_data=checkpointer.restore("/home/guorui/IDVE/expert_data/common_road_policy")
        dataset={}
        dataset["actions"] = jnp.array(expert_data["action"]).astype(jnp.float32)
        dataset["observations"] = jnp.array(expert_data["traj_obs"]).astype(jnp.float32)
        dataset["next_observations"] = jnp.array(expert_data["traj_next_obs"]).astype(jnp.float32)
        dataset["terminals"] = jnp.array(expert_data["dones"]).astype(jnp.float32)
        dataset["rewards"] = jnp.array(expert_data["rewards"]).astype(jnp.float32)
        dataset["index"] = jnp.array(expert_data["index"]).astype(jnp.float32)
        return dataset

