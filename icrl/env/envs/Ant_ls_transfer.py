import jax.numpy as jnp
import jax
import numpy as np
import gym as ogym

from flax.training import orbax_utils
import orbax.checkpoint
from icrl.common.venv_wrappers import EnvWrapper
import flax
env_id="Ant-v4"


def cost_function(next_obs, reward, next_done, next_truncated, info):
    if 'true_obs' in info.keys():
        return  (info['true_obs'][:,13]>1.0)
    else:
        return (next_obs[:,13]>1.0)


# theta=-jnp.pi/6

# def reward_function(next_obs, reward, next_done, next_truncated, info):
#     reward_v=jnp.cos(theta)*info['x_velocity']+jnp.sin(theta)*(info['y_velocity'])
#     return (info['reward_ctrl']+info['reward_survive']+info['reward_contact']+reward_v).astype(jnp.float32)

# @flax.struct.dataclass
# class RewardWrapper(EnvWrapper):
#     def recv(self,ret) :
#         next_obs, reward, next_done,next_truncated, info= ret
#         reward=reward_function(next_obs, reward, next_done, next_truncated, info)
#         return self, (next_obs, reward, next_done,next_truncated, info)

# def get_wrappers(envs):
#     return [RewardWrapper()]

def offline_dataset():
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("tmp/buffer/Ant_ls/Ant_ls__new_sac_sample__1__1703215629")
    dataset={}
    dataset["observations"]=data[0][:100000,0]
    dataset["next_observations"]=data[1][:100000,0]
    dataset["actions"]=data[2][:100000,0]
    dataset["infos"]=jax.tree.map(
            lambda x:x[1:,0], data[5]
        )
    dataset["rewards"]=(data[5]['reward_ctrl']+data[5]['reward_survive']+data[5]['reward_contact']+data[5]['x_velocity']).astype(jnp.float32)
    dataset["terminals"]=data[3][:100000,0]
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    dataset["index"]=jnp.array(data[6])

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("expert_data/Ant_ls_transfer_limited")
    dataset1={}
    dataset1["observations"]=jnp.array(data["traj_obs"]).astype(jnp.float32)
    dataset1["next_observations"]=jnp.array(data["traj_next_obs"]).astype(jnp.float32)
    dataset1["actions"]=jnp.array(data["action"]).astype(jnp.float32)
    dataset1["infos"]=data["infos"]
    dataset1["rewards"]=(data["infos"]['reward_ctrl']+data["infos"]['reward_survive']+data["infos"]['reward_contact']+data["infos"]['x_velocity']).astype(jnp.float32)
    dataset1["terminals"]=jnp.array(data["dones"]).astype(jnp.float32)
    dataset1["timeouts"]=jnp.zeros_like(dataset1["terminals"])
    dataset1["index"]=jnp.array(data["index"]).astype(jnp.float32)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore("expert_data/Ant_ls_transfer_unlimited")
    dataset2={}
    dataset2["observations"]=jnp.array(data["traj_obs"]).astype(jnp.float32)
    dataset2["next_observations"]=jnp.array(data["traj_next_obs"]).astype(jnp.float32)
    dataset2["actions"]=jnp.array(data["action"]).astype(jnp.float32)
    dataset2["infos"]=data["infos"]
    dataset2["rewards"]=(data["infos"]['reward_ctrl']+data["infos"]['reward_survive']+data["infos"]['reward_contact']+data["infos"]['x_velocity']).astype(jnp.float32)
    dataset2["terminals"]=jnp.array(data["dones"]).astype(jnp.float32)
    dataset2["timeouts"]=jnp.zeros_like(dataset2["terminals"])
    dataset2["index"]=jnp.array(data["index"]).astype(jnp.float32)

    return jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]),  dataset1,dataset2
        )
    # return jax.tree.map(
    #         lambda x, y,z: jnp.concatenate([x, y,z]), dataset, dataset1,dataset2
    #     )
