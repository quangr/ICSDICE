env_id="commonroad"
from icrl.common.DataReader import get_dataset
import numpy as np
def cost_function(next_obs, reward, next_done, next_truncated, info):
        return np.linalg.norm(next_obs[:,6:8],axis=-1)>40



def offline_dataset():
    dataset1=get_dataset("tmp/buffer/Ant_ls/Ant_ls__new_sac_sample_common_road__2__1711453802/",0,200000)
    return dataset1