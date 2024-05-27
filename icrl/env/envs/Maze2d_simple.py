import jax.numpy as jnp
env_id="Maze2d_simple"
def cost_function(next_obs, reward, next_done, next_truncated, info):
    return info["i"]==2 and (info["j"]==1)
