import jax.numpy as jnp
import orbax.checkpoint
import jax

def get_dataset(path,start=0,end=100000):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data=checkpointer.restore(path)
    dataset={}
    dataset["observations"]=data[0][start:end,0]
    dataset["next_observations"]=data[1][start:end,0]
    dataset["actions"]=data[2][start:end,0]
    dataset["rewards"]=data[4][start:end,0]
    dataset["terminals"]=data[3][start:end,0]
    dataset["infos"]=jax.tree.map(
            lambda x:x[start+1:end+1,0], data[5]
        )
    dataset["timeouts"]=jnp.zeros_like(dataset["terminals"])
    dataset["index"]=jnp.array(data[6][start:end])
    return dataset
