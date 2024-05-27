
import flax
import jax.numpy as jnp
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import flax.linen as nn
import numpy as np

import collections

AgentState = collections.namedtuple(
    "AgentState",
    [
        "reward_qf_state",
        "reward_vf_state",
        "key",
    ],
)


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict = None

class Critic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        return x


class VectorCritic(nn.Module):
    n_units: int = 256
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            n_units=self.n_units,
        )(obs, action)
        return q_values


class ValueCritic(nn.Module):
    n_units: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(self.n_units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(1, kernel_init=orthogonal((0.01)*jnp.sqrt(2)), bias_init=constant(0.0))(x)
        return x

def concatenated_indices(input_array):
    # Find the indices where the value is 1
    assert(input_array[0]==1)
    indices = np.where(input_array == 1)[0]
    last_indices = np.concatenate([indices[1:], [len(input_array)]])
    # Use np.arange and concatenate
    result = np.concatenate([np.arange(l - r) for r, l in zip(indices, last_indices)])

    return result,list(zip(indices, last_indices))


@flax.struct.dataclass
class BatchData:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    index: np.ndarray
    costs: np.ndarray
    init_observations: np.ndarray
