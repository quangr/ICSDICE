import flax
import jax
import jax.numpy as jnp
@flax.struct.dataclass
class ReplayBuffer:
    obs: jnp.array
    actions: jnp.array
    rewards: jnp.array
    next_obs: jnp.array
    dones: jnp.array
    buffer_size:int
    count:int

    @classmethod
    def create(cls, buffer_size,obs_shape,action_shape):
        obs=jnp.empty((buffer_size,*obs_shape), dtype=jnp.float32)
        next_obs=jnp.empty((buffer_size,*obs_shape), dtype=jnp.float32)
        actions=jnp.empty((buffer_size,*action_shape), dtype=jnp.float32)
        jnp.empty((buffer_size,*obs_shape), dtype=jnp.float32)
        dones=jnp.empty((buffer_size,*obs_shape), dtype=jnp.float32)
        return cls(
            buffer_size=buffer_size,
            count=0,
            obs=obs,
            next_obs=next_obs,
            actions=actions,
            dones=dones,
            rewards=dones
        )


    @property
    def size(self):
        # WARN: do not use __len__ here! It will use len of the dataclass, i.e. number of fields.
        return self.count

    def sample_batch(self, key: jax.random.PRNGKey, batch_size: int):
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size)
        batch = jax.tree.map(lambda arr: arr[indices], self)
        return batch
