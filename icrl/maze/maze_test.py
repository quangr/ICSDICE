# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import jax
import numpy as np
import gymnasium as gym
import orbax.checkpoint

import numpy as np
import matplotlib.pyplot as plt
import optax
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState
import orbax.checkpoint

from icrl.maze.utility import (
    RLTrainState,
    VectorCritic,
    ValueCritic,
    concatenated_indices,
    AgentState,
    BatchData,
)

env_id = "Maze2d_simple"
env = gym.make(env_id)

key = jax.random.PRNGKey(0)
key, actor_key, qf_key, ent_vkey = jax.random.split(key, 4)

obs = jnp.array([env.observation_space.sample()])
action = jnp.array([env.action_space.sample()])

vf = ValueCritic()
vf_state = TrainState.create(
    apply_fn=vf.apply,
    params=vf.init(actor_key, obs),
    tx=optax.adam(learning_rate=3e-4),
)

reward_vf_state = TrainState.create(
    apply_fn=vf.apply,
    params=vf.init(actor_key, obs),
    tx=optax.adam(learning_rate=3e-4),
)


qf = VectorCritic(n_critics=1)

qf_state = RLTrainState.create(
    apply_fn=qf.apply,
    params=qf.init({"params": qf_key}, obs, action),
    target_params=qf.init({"params": qf_key}, obs, action),
    tx=optax.adam(learning_rate=3e-4),
)

reward_qf_state = RLTrainState.create(
    apply_fn=qf.apply,
    params=qf.init({"params": qf_key}, obs, action),
    target_params=qf.init({"params": qf_key}, obs, action),
    tx=optax.adam(learning_rate=3e-4),
)

# %%
# def get_dataset(setting):
#     checkpointer = orbax.checkpoint.PyTreeCheckpointer()
#     data = checkpointer.restore(
#         "/home/guorui/jax-rl/tmp/buffer/Maze2d_simple/Maze2d_simple__new_sac_reward_shaping__1__1700637466/"
#     )
#     dataset = {}
#     if setting == "single":
#         start_index = 850 + 869
#         end_index = start_index + 160
#     elif setting == "multi":
#         start_index = 850 + 300 + 1
#         end_index = start_index + 729 - 1
#     else:
#         start_index = 0
#         # end_index = 50000
#         end_index = 20000
#     dataset["observations"] = jnp.array(
#         data[0][start_index:end_index, 0, :4].astype(jnp.float32)
#     )
#     dataset["next_observations"] = jnp.array(
#         data[1][start_index:end_index, 0, :4].astype(jnp.float32)
#     )
#     dataset["actions"] = jnp.array(data[2][start_index:end_index, 0])
#     dataset["rewards"] = (
#         jnp.array(data[4][start_index:end_index, 0].clip(0)) - 100
#     ) / 100
#     dataset["infos"] = data[5]
#     dataset["terminals"] = jnp.array(data[3][start_index:end_index, 0])
#     dataset["timeouts"] = jnp.zeros_like(dataset["terminals"])
#     startpoint = (-1, -1)
#     dataset["index"] = jnp.array(
#         (dataset["observations"][:, 0] == startpoint[0])
#         & (dataset["observations"][:, 1] == startpoint[1])
#     )
#     dataset["index"] = jnp.array(concatenated_indices(dataset["index"]))
#     dataset["init_eff"] = 1 / (dataset["index"] == 0).mean()
#     return dataset


# data_setting = "all"
# dataset = get_dataset(data_setting)

# %%
def get_data(index_list):
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    data = checkpointer.restore(
        "tmp/buffer/Maze2d_simple/Maze2d_simple__new_sac_reward_shaping__1__1700637466/"
    )
    dataset = {}
    start_index = 0
    end_index = 50000
    dataset["observations"] = jnp.array(
        data[0][start_index:end_index, 0, :4].astype(jnp.float32)
    )
    startpoint = (-1, -1)
    dataset["index"] = jnp.array(
        (dataset["observations"][:, 0] == startpoint[0])
        & (dataset["observations"][:, 1] == startpoint[1])
    )

    import random
    intervals=concatenated_indices(dataset["index"])[1]
    sample_interval = [intervals[i] for i in index_list]
    print(index_list)
    selected_index = np.concatenate(
        [np.arange(start, end) for (start, end) in sample_interval]
    )
    dataset["observations"] = jnp.array(
        data[0][selected_index, 0, :4].astype(jnp.float32)
    )
    dataset["next_observations"] = jnp.array(
        data[1][selected_index, 0, :4].astype(jnp.float32)
    )
    dataset["actions"] = jnp.array(data[2][selected_index, 0])
    dataset["rewards"] = (jnp.array(data[4][selected_index, 0].clip(0)) - 100) / 100
    dataset["infos"] = data[5]
    dataset["terminals"] = jnp.array(data[3][selected_index, 0])
    dataset["timeouts"] = jnp.zeros_like(dataset["terminals"])
    startpoint = (-1, -1)
    dataset["index"] = jnp.array(
        (dataset["observations"][:, 0] == startpoint[0])
        & (dataset["observations"][:, 1] == startpoint[1])
    )
    dataset["index"] = jnp.array(concatenated_indices(dataset["index"])[0])
    dataset["init_eff"] = 1 / (dataset["index"] == 0).mean()
    dataset["init_obs"] = dataset["observations"][dataset["index"] == 0]
    return dataset

dataset = get_data([193, 5,   8,63 ])


# %%
def sample_batch(key):
    sample_key, key = jax.random.split(key, 2)
    expert_indice = jax.random.randint(
        sample_key, minval=0, maxval=len(dataset["observations"]), shape=(256,)
    )
    Policy_Batch = BatchData(
        dataset["observations"][expert_indice],
        dataset["actions"][expert_indice],
        dataset["next_observations"][expert_indice],
        dataset["rewards"][expert_indice],
        dataset["terminals"][expert_indice],
        expert_indice,
        expert_indice,
        expert_indice,
    )

    return Policy_Batch, key

# %%
# def omega_star(y):
#     return (y>0)*(y / 2 + 1)+(y<=0)*(jnp.exp(y))
# def fp_star(y):
#     return (y>0)*(y*(y+4)/4)+(y<=0)*(jnp.exp(y)-1)


def omega_star(y):
    return (y > -2) * (y / 2 + 1)


def fp_star(y):
    return (y > -2) * (y * (y + 4) / 4)


def cost_region(x, y):
    return (x < -0.2) & (y < 0.3) & (y > -0.3)


def train_baseline(gamma=0.99, alpha=0.05, init_eff=1):
    def reward_train_step(agent_state):
        callback_log = {}

        def update_reward_value_critic(
            Policy_Batch: BatchData,
            reward_qf_state: RLTrainState,
            reward_vf_state: RLTrainState,
            key: jax.Array,
        ):
            Batch = Policy_Batch
            next_feasible = 1 - cost_region(
                Batch.next_observations[:, 0],
                Batch.next_observations[:, 1],
            )

            def mse_loss(params):
                current_reward_V = vf.apply(params, Batch.observations).reshape(-1)
                next_reward_V = vf.apply(params, Batch.next_observations).reshape(-1)
                y = gamma * next_reward_V - current_reward_V + Batch.rewards
                index = dataset["index"][Batch.index]
                loss = (
                    (1 - gamma) * init_eff * ((index == 0) * current_reward_V).mean()
                    + alpha
                    * (next_feasible * (gamma**index) * fp_star(y / alpha)).mean()
                    + alpha
                    * (
                        Batch.dones
                        * (gamma ** (index + 1))
                        / (1 - gamma)
                        * fp_star((gamma * next_reward_V - next_reward_V) / alpha)
                    ).mean()
                )
                callback_log["terminal_reg"] = jax.lax.stop_gradient(
                    (Batch.dones).sum()
                )
                return loss

            rewad_vf_loss_value, grads = jax.value_and_grad(mse_loss)(
                reward_vf_state.params
            )
            callback_log["rewad_vf_loss_value"] = rewad_vf_loss_value
            reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
            return reward_vf_state, key

        (
            reward_qf_state,
            reward_vf_state,
            key,
        ) = agent_state
        Policy_Batch, key = sample_batch(key)
        reward_vf_state, key = update_reward_value_critic(
            Policy_Batch,
            reward_qf_state,
            reward_vf_state,
            key,
        )
        return (
            AgentState(
                reward_qf_state,
                reward_vf_state,
                key,
            ),
            callback_log,
        )

    agent_state = AgentState(
        reward_qf_state,
        reward_vf_state,
        key,
    )

    _reward_train_step = jax.jit(reward_train_step)

    def reward_train_step_body(carry, step):
        agentstate = carry
        agentstate, callback_log = _reward_train_step(agentstate)
        return agentstate, (callback_log, agentstate.reward_vf_state)

    for i in range(200):
        agent_state, (reward_callback_log, reward_vf_states) = jax.lax.scan(
            reward_train_step_body,
            (agent_state),
            (),
            length=1000,
        )
    return agent_state, reward_callback_log


def plot_value(agent_state, alpha, gamma):
    next_v = vf.apply(
        agent_state.reward_vf_state.params, dataset["next_observations"]
    ).reshape(-1)
    v = vf.apply(agent_state.reward_vf_state.params, dataset["observations"]).reshape(
        -1
    )
    tdv = (gamma * next_v + dataset["rewards"] - v) / (2 * alpha)
    print(next_v[-2:])
    return tdv
# %%
alphas = [0.02,  0.001, 0.0005, 0.0001]
tdvs = {}
plots = {}
for i, alpha in enumerate(alphas):
    gamma = 0.99
    agent_state_without_q, reward_callback_log_without_q = train_baseline(
        gamma=gamma, alpha=alpha, init_eff=dataset["init_eff"]
    )
    tdvs[alpha] = plot_value(agent_state_without_q, alpha, gamma)
limits = np.linspace(-2, -1, 3)
plot_alphas = alphas
fig, axes = plt.subplots(
    len(limits), len(plot_alphas), figsize=(3 * len(plot_alphas), 2 * len(limits))
)
for i, limit in enumerate(limits):
    for j, alpha in enumerate(plot_alphas):
        sc = axes[i, j].scatter(
            dataset["observations"][:, 0],
            dataset["observations"][:, 1],
            c=(tdvs[alpha] > limit),
            vmin=0,
            vmax=1,
            cmap="Greys",
            s=1,
            rasterized=True,
        )
for ax, col in zip(axes[:, 0], limits):
    ax.set_ylabel("limit=" + str(col), rotation=0)
    ax.yaxis.set_label_coords(-0.4, 0.5)
for ax, row in zip(axes[0], plot_alphas):
    ax.set_title(r"$\alpha$=" + str(row))

# %%


# %%
alphas = [1, 0.2, 0.05, 0.005]
tdvs = {}
plots = {}
for i, alpha in enumerate(alphas):
    gamma = 0.99
    agent_state_without_q, reward_callback_log_without_q = train_baseline(
        gamma=gamma, alpha=alpha, init_eff=1
    )
    tdvs[alpha], plots[alpha] = plot_value(agent_state_without_q, alpha, gamma)
limits = np.linspace(-2, -1, 3)
plot_alphas = alphas
fig, axes = plt.subplots(
    len(limits), len(plot_alphas), figsize=(3 * len(plot_alphas), 2 * len(limits))
)
for i, limit in enumerate(limits):
    for j, alpha in enumerate(plot_alphas):
        sc = axes[i, j].scatter(
            dataset["observations"][:, 0],
            dataset["observations"][:, 1],
            c=(tdvs[alpha] > limit),
            vmin=0,
            vmax=1,
            cmap="Greys",
            s=1,
            rasterized=True,
        )
for ax, col in zip(axes[:, 0], limits):
    ax.set_ylabel("limit=" + str(col), rotation=0)
    ax.yaxis.set_label_coords(-0.4, 0.5)
for ax, row in zip(axes[0], plot_alphas):
    ax.set_title(r"$\alpha$=" + str(row))


