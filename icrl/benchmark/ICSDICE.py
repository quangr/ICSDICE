import argparse
import os
import random
import time
from distutils.util import strtobool
from functools import partial
from typing import Sequence
import flax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import orbax_utils
import orbax.checkpoint
import collections

from flax.training.train_state import TrainState

from icrl.env.cost_env import make_cost_env_no_xla


@flax.struct.dataclass
class BatchData:
    observations: np.ndarray
    actions: np.ndarray
    next_observations: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    time_step: np.ndarray


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="paper",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--eval-freq", type=int, default=-1,
        help="evaluate the agent every `eval_freq` steps (if negative, no evaluation)")
    parser.add_argument("--n-eval-episodes", type=int, default=10,
        help="number of episodes to use for evaluation")
    parser.add_argument("--n-eval-envs", type=int, default=5,
        help="number of environments for evaluation")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="HalfCheetahWithObstacle",
        help="the id of the environment to be used")
    parser.add_argument("--expert-data", type=str,
        help="the name of the expert data file")
    parser.add_argument("--update-period", type=int, default=10000)
    parser.add_argument("--expert-ratio", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1,
        help="the reward alpha parameter")
    parser.add_argument("--total-timesteps", type=int, default=100000,
        help="total timesteps of the experiments")
    parser.add_argument("--buffer-size", type=int, default=int(1e6),
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--l1-ratio", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=0.005,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--policy-lr", type=float, default=3e-4,
        help="the learning rate of the policy network optimizer")
    parser.add_argument("--q-lr", type=float, default=3e-4,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--sample-num", type=int, default=50,
        help="the learning rate of the Q network network optimizer")
    parser.add_argument("--policy-frequency", type=int, default=1,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--cost-l2", type=float, default=0.01,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--cost-limit", type=float, default=0.9,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--target-network-frequency", type=int, default=1, # Denis Yarats' implementation delays this by 2.
        help="the frequency of updates for the target nerworks")
    parser.add_argument("--alpha", type=float, default=0.0005,
        help="entropy regularization coefficient")
    parser.add_argument('--debug', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--mean-V-update', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,)
    parser.add_argument('--update-buffer', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--update-V', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument('--stable-update', type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,)
    parser.add_argument('--use-q', type=lambda x:bool(strtobool(x)), default=True, nargs="?", const=False,)
    parser.add_argument("--autotune", type=lambda x:bool(strtobool(x)), default=False, nargs="?", const=False,
        help="automatic tuning of the entropy coefficient")
    args = parser.parse_args()
    # fmt: on
    if args.expert_data is None:
        args.expert_data = "dataset/" + args.env_id+"/expert"
    return args


class Critic(nn.Module):
    n_units: int = 256

    @nn.compact
    def __call__(self, x: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([x, action], -1)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(
            1,
            1,
            kernel_init=orthogonal((0.01)),
            bias_init=constant(0.0),
        )(x)
        return x


class ValueCritic(nn.Module):
    n_units: int = 512

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(
            self.n_units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.n_units, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
        x = nn.relu(x)
        x = nn.Dense(
            1, kernel_init=orthogonal((0.01) * jnp.sqrt(2)), bias_init=constant(0.0)
        )(x)
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


class Actor(nn.Module):
    action_dim: Sequence[int]
    n_units: int = 256
    log_std_min: float = -20
    log_std_max: float = 2

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        x = nn.Dense(self.n_units)(x)
        x = nn.relu(x)
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std


class RLTrainState(TrainState):
    target_params: flax.core.FrozenDict = None


@partial(jax.jit, static_argnames="actor")
def sample_action(
    actor: Actor,
    actor_state: TrainState,
    observations: jnp.ndarray,
    key: jax.Array,
) -> jnp.array:
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor.apply(actor_state.params, observations)
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(subkey, shape=mean.shape)
    action = jnp.tanh(gaussian_action)
    return action, key


@partial(jax.jit, static_argnames="actor")
def sample_action_mean(
    actor: Actor,
    actor_state: TrainState,
    observations: jnp.ndarray,
    key: jax.Array,
) -> jnp.array:
    key, subkey = jax.random.split(key, 2)
    mean, log_std = actor.apply(actor_state.params, observations)
    gaussian_action = mean
    action = jnp.tanh(gaussian_action)
    return action, key


@jax.jit
def sample_action_and_log_prob(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    subkey: jax.Array,
):
    action_std = jnp.exp(log_std)
    gaussian_action = mean + action_std * jax.random.normal(
        subkey, shape=mean.shape, dtype=jnp.float32
    )
    log_prob = (
        -0.5 * ((gaussian_action - mean) / action_std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    log_prob = log_prob.sum(axis=1)
    action = jnp.tanh(gaussian_action)
    log_prob -= jnp.sum(jnp.log((1 - action**2) + 1e-6), 1)
    return action, log_prob


@jax.jit
def action_log_prob(
    action: jnp.ndarray,
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
):
    action = action.clip(-1 + 1e-6, 1 - 1e-6)
    action_std = jnp.exp(log_std)
    gaussian_action = jnp.arctanh(action)
    log_prob = (
        -0.5 * ((gaussian_action - mean) / action_std) ** 2
        - 0.5 * jnp.log(2.0 * jnp.pi)
        - log_std
    )
    log_prob -= jnp.log((1 - action**2) + 1e-6)
    return log_prob.sum(-1)


@partial(jax.jit, static_argnames="actor")
def select_action(
    actor: Actor, actor_state: TrainState, observations: jnp.ndarray
) -> jnp.array:
    return actor.apply(actor_state.params, observations)[0]


def scale_action(action_space, action) -> np.ndarray:
    """
    Rescale the action from [low, high] to [-1, 1]
    (no need for symmetric action space)

    :param action: Action to scale
    :return: Scaled action
    """
    low, high = action_space.low, action_space.high
    return 2.0 * ((action - low) / (high - low)) - 1.0


def unscale_action(action_space, scaled_action) -> np.ndarray:
    """
    Rescale the action from [-1, 1] to [low, high]
    (no need for symmetric action space)

    :param scaled_action: Action to un-scale
    """
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


class EntropyCoef(nn.Module):
    ent_coef_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        log_ent_coef = self.param(
            "log_ent_coef",
            init_fn=lambda key: jnp.full((), jnp.log(self.ent_coef_init)),
        )
        return jnp.exp(log_ent_coef)


def l2_loss(x):
    return (x**2).mean()


@jax.jit
def soft_update(tau: float, qf_state: RLTrainState) -> RLTrainState:
    qf_state = qf_state.replace(
        target_params=optax.incremental_update(
            qf_state.params, qf_state.target_params, tau
        )
    )
    return qf_state


def l2_loss(x):
    return (x**2).mean()


@jax.jit
def get_log_prob(params, actions, observations):
    mean, log_std = actor.apply(params, observations)
    log_prob = action_log_prob(actions, mean, log_std)
    return log_prob.mean()


def sample_batch(key):
    sample_key, key = jax.random.split(key, 2)
    policy_index = jax.random.randint(
        sample_key,
        minval=0,
        maxval=len(policy_obs),
        shape=(2 * args.batch_size - int(args.batch_size * args.beta),),
    )
    Policy_Batch = BatchData(
        policy_obs[policy_index],
        policy_actions[policy_index],
        policy_next_obs[policy_index],
        policy_rewards[policy_index],
        policy_dones[policy_index],
        policy_timestep[policy_index],
    )
    sample_key, key = jax.random.split(key, 2)
    expert_index = jax.random.randint(
        sample_key,
        minval=0,
        maxval=len(expert_obs),
        shape=(int(args.batch_size * args.beta),),
    )
    Expert_Batch = BatchData(
        expert_obs[expert_index],
        expert_actions[expert_index],
        expert_next_obs[expert_index],
        expert_rewards[expert_index],
        expert_dones[expert_index],
        expert_timestep[expert_index],
    )

    return Policy_Batch, Expert_Batch, (policy_index, expert_index)


def reward_train_step(agent_state):
    callback_log = {}

    def update_reward_value_critic(
        Expert_Batch: BatchData,
        Policy_Batch: BatchData,
        vf_state: RLTrainState,
        reward_vf_state: RLTrainState,
    ):
        Batch = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        next_cost = vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)
        policy_length = len(Policy_Batch.observations)

        def mse_loss(params):
            current_reward_V = vf.apply(params, Batch.observations).reshape(-1)
            next_reward_V = vf.apply(params, Batch.next_observations).reshape(-1)
            mask = next_cost < cost_limit
            y = gamma * next_reward_V + Batch.rewards - current_reward_V
            if args.debug:
                callback_log["next_cost"] = jax.lax.stop_gradient((next_cost).mean())
                callback_log["d_positive"] = jax.lax.stop_gradient(
                    (y / (2 * alpha) + 1 > 0).sum()
                )
                callback_log["mask_sum"] = mask.sum()

            loss = (
                (1 - gamma)
                * init_eff
                * ((Batch.time_step == 0) * current_reward_V).mean()
                # + ((1 - mask) * (gamma**Batch.time_step) * next_reward_V).mean() * 0.1
                + alpha
                * (mask * (gamma**Batch.time_step) * fp_star(y / alpha)).sum()
                / (mask.sum() + (mask * Batch.dones).sum())
                + alpha
                * (
                    mask
                    * Batch.dones
                    * (gamma ** (Batch.time_step + 1))
                    / (1 - gamma)
                    * fp_star((gamma * next_reward_V - next_reward_V) / alpha)
                ).sum()
                / (mask.sum() + (mask * Batch.dones).sum())
            )
            return loss, (
                current_reward_V[:policy_length].mean(),
                current_reward_V[policy_length:].mean(),
            )

        (
            rewad_vf_loss_value,
            (
                policy_reward_vf_values,
                expert_reward_vf_values,
            ),
        ), grads = jax.value_and_grad(mse_loss, has_aux=True)(reward_vf_state.params)
        reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
        if args.debug:
            callback_log["reward_vf_loss_value"] = rewad_vf_loss_value
            callback_log["policy_reward_vf_values"] = policy_reward_vf_values
            callback_log["expert_reward_vf_values"] = expert_reward_vf_values
        return reward_vf_state

    def update_actor(
        actor_state: RLTrainState,
        qf_state: RLTrainState,
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
        reward_qf_state: RLTrainState,
        reward_vf_state: RLTrainState,
    ):
        Batch = jax.tree.map(
            lambda x, y: jnp.concatenate([x, y]), Policy_Batch, Expert_Batch
        )
        current_reward_V = vf.apply(reward_vf_state.params, Batch.observations).reshape(
            -1
        )
        next_reward_V = vf.apply(reward_vf_state.params, Batch.observations).reshape(-1)
        next_cost_pi = vf.apply(
            vf_state.params, Batch.next_observations * single_mask
        ).reshape(-1)

        def actor_loss(params):
            mean, log_std = actor.apply(params, Batch.observations)
            log_prob = action_log_prob(Batch.actions, mean, log_std)
            y = gamma * next_reward_V + Batch.rewards - current_reward_V
            mask = next_cost_pi < cost_limit

            if args.debug:
                callback_log["policy_td_mean"] = jax.lax.stop_gradient(
                    y[: len(Policy_Batch.observations)].mean()
                )
                callback_log["expert_td_mean"] = jax.lax.stop_gradient(
                    y[len(Policy_Batch.observations) :].mean()
                )
                callback_log["mask_logp"] = jax.lax.stop_gradient(
                    ((1 - mask) * log_prob).mean()
                )
            loss = -(mask * jnp.exp(y).clip(max=10.0) * log_prob).mean()
            # loss = -(jnp.exp(-(1 - mask) * 1000 + y) * log_prob).mean()
            return loss, -log_prob.mean()

        (actor_loss_value, entropy), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)
        callback_log["actor_loss_value"] = actor_loss_value
        callback_log["entropy"] = entropy
        return actor_state

    (
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    ) = agent_state
    key, subkey = jax.random.split(key, 2)

    Policy_Batch, Expert_Batch, _ = sample_batch(subkey)
    reward_vf_state = update_reward_value_critic(
        Expert_Batch,
        Policy_Batch,
        vf_state,
        reward_vf_state,
    )
    actor_state = update_actor(
        actor_state,
        qf_state,
        vf_state,
        Policy_Batch,
        Expert_Batch,
        reward_qf_state,
        reward_vf_state,
    )

    return (
        AgentState(
            actor_state,
            vf_state,
            qf_state,
            reward_qf_state,
            reward_vf_state,
            key,
        ),
        callback_log,
    )


def train_step(agent_state, weight_buffer):
    callback_log = {}

    def update_value_critic(
        vf_state: RLTrainState,
        Policy_Batch: BatchData,
        Expert_Batch: BatchData,
    ):
        weight_p = weight_buffer.weights[weight_index, policy_index]

        def mse_loss(params):
            expert_cost = vf.apply(
                params, Expert_Batch.observations * single_mask
            ).reshape(-1)
            policy_cost = vf.apply(
                params, Policy_Batch.observations * single_mask
            ).reshape(-1)
            learning_loss = (
                ((expert_cost) ** 2).mean()
                + ((weight_p > 0) * (policy_cost - 1) ** 2).mean()
                + args.cost_l2 * ((policy_cost) ** 2).mean()
            )

            loss = (
                learning_loss
                # + jnp.abs(expert_cost).mean() * 1000
                # + jnp.abs(policy_cost).mean() * 1000
                # - policy_next_V.clip(max=0).mean() *100
                # - expert_next_V.clip(max=0).mean() *100
            )
            return loss

        vf_loss_value, grads = jax.value_and_grad(mse_loss)(vf_state.params)
        vf_state = vf_state.apply_gradients(grads=grads)

        callback_log["vf_loss_value"] = vf_loss_value

        return vf_state

    (
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    ) = agent_state
    key, subkey = jax.random.split(key, 2)

    Policy_Batch, Expert_Batch, (policy_index, expert_index) = sample_batch(subkey)
    sample_key, key = jax.random.split(key, 2)
    weight_index = jax.random.randint(
        sample_key, minval=0, maxval=weight_buffer.count, shape=(1,)
    )
    vf_state = update_value_critic(
        vf_state,
        Policy_Batch,
        Expert_Batch,
    )

    return (
        AgentState(
            actor_state,
            vf_state,
            qf_state,
            reward_qf_state,
            reward_vf_state,
            key,
        ),
        callback_log,
    )


AgentState = collections.namedtuple(
    "AgentState",
    [
        "actor_state",
        "vf_state",
        "qf_state",
        "reward_qf_state",
        "reward_vf_state",
        "key",
    ],
)


def eval_policy(envs, agentstate, global_step, round=10):
    reward_deque = []
    cost_deque = []
    length_deque = []
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    try:
        _, (obs, info) = envs.reset()
    except ValueError:  # Catch the exception if envs.reset() doesn't return a tuple
        obs, info = envs.reset()
    key = agentstate.key
    for i in range(10):
        while True:
            obs = obs.astype(np.float32)
            actions, key = sample_action_mean(
                actor,
                agentstate.actor_state,
                normalize_datas(obs, obs_means, obs_stds),
                key,
            )
            input_actions = unscale_action(envs.action_space, actions.__array__())
            (next_obs, rewards, dones, truncated, infos) = envs.step(input_actions)
            costs = cost_function(next_obs, rewards, dones, truncated, infos)
            next_obs = next_obs.astype(np.float32)

            obs = next_obs
            cum_reward += rewards
            cum_cost += costs
            lengths += 1
            if dones.any() or truncated.any():
                reward_deque.append(cum_reward)
                cost_deque.append(cum_cost)
                length_deque.append(lengths)
                cum_reward = 0
                cum_cost = 0
                lengths = 0
                try:
                    _, (obs, info) = envs.reset()
                except (
                    ValueError
                ):  # Catch the exception if envs.reset() doesn't return a tuple
                    obs, info = envs.reset()
                obs = obs.astype(np.float32)
                break
    log_data = {
        "charts/mean_return": np.average(reward_deque),
        "charts/mean_cost": np.average(cost_deque),
        "charts/mean_length": np.average(length_deque),
    }
    for k, v in log_data.items():
        print(k, v, global_step)
    # Log the data if args.track is True
    if args.track:
        # Assuming you have imported the required libraries (e.g., wandb)
        wandb.log({"global_step": global_step, **log_data})


def f_div(x):
    return (x - 1) ** 2


def omega_star(y):
    return (y > -2) * (y / 2 + 1)


def fp_star(y):
    return (y > -2) * (y * (y + 4) / 4)


def split_data(expert_data):
    expert_actions = jnp.array(expert_data["action"]).astype(jnp.float32)
    expert_obs = jnp.array(expert_data["traj_obs"]).astype(jnp.float32)
    expert_next_obs = jnp.array(expert_data["traj_next_obs"]).astype(jnp.float32)
    expert_dones = jnp.array(expert_data["dones"]).astype(jnp.float32)
    expert_rewards = jnp.array(expert_data["rewards"]).astype(jnp.float32)
    expert_timestep = jnp.array(expert_data["index"]).astype(jnp.float32)
    return (
        expert_obs,
        expert_next_obs,
        expert_actions,
        expert_rewards,
        expert_dones,
        expert_timestep,
    )


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time()*1000)}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
            tags=[os.environ["RUN_TAG"]] if os.environ.get("RUN_TAG") else [],
        )
    mask_coff = -0.01
    gamma = args.gamma
    alpha = args.alpha
    cost_limit = args.cost_limit

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    # Use a separate key, so running with/without eval doesn't affect the results
    eval_key = jax.random.PRNGKey(args.seed)
    envs, cost_function, env_module = make_cost_env_no_xla(
        args.env_id, 1, args.seed, get_module=True
    )

    # Create networks
    key, actor_key, qf_key, ent_key = jax.random.split(key, 4)

    obs = jnp.array([envs.single_observation_space.sample()]).astype(jnp.float32)
    action = jnp.array([envs.single_action_space.sample()]).astype(jnp.float32)
    if hasattr(env_module, "single_mask"):
        single_mask = env_module.single_mask
    else:
        single_mask = np.ones(len(obs[0]))
    actor = Actor(action_dim=np.prod(envs.single_action_space.shape))

    update_period = args.update_period
    step_per_scan = 1000
    update_preiod = int((args.update_period) / step_per_scan)
    total_step = int((args.total_timesteps) / step_per_scan)

    actor_optimiser = optax.adam(learning_rate=args.policy_lr)
    q_optimiser = optax.adam(learning_rate=args.q_lr)
    v_optimiser = optax.adamw(learning_rate=args.q_lr)

    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=actor_optimiser,
    )

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    _expert_data = checkpointer.restore(args.expert_data)
    (
        expert_obs,
        expert_next_obs,
        expert_actions,
        expert_rewards,
        expert_dones,
        expert_timestep,
    ) = split_data(_expert_data)
    random_indice = np.random.permutation(len(_expert_data["traj_obs"]))
    other_indice = random_indice[
        int(args.expert_ratio * len(_expert_data["traj_obs"])) :
    ]
    random_indice = random_indice[
        : int(args.expert_ratio * len(_expert_data["traj_obs"]))
    ]
    (
        expert_obs,
        expert_next_obs,
        expert_actions,
        expert_rewards,
        expert_dones,
        expert_timestep,
    ) = jax.tree.map(
        lambda x: x[random_indice],
        (
            expert_obs,
            expert_next_obs,
            expert_actions,
            expert_rewards,
            expert_dones,
            expert_timestep,
        ),
    )
    (
        other_expert_obs,
        other_expert_next_obs,
        other_expert_actions,
        other_expert_rewards,
        other_expert_dones,
        other_expert_timestep,
    ) = jax.tree.map(
        lambda x: x[other_indice],
        (
            expert_obs,
            expert_next_obs,
            expert_actions,
            expert_rewards,
            expert_dones,
            expert_timestep,
        ),
    )
    vf = ValueCritic()
    vf.apply = jax.jit(vf.apply)
    vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=v_optimiser,
    )

    qf = VectorCritic(n_critics=1)

    qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=q_optimiser,
    )

    qf = VectorCritic(n_critics=1)

    reward_qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=q_optimiser,
    )

    reward_vf = ValueCritic()
    reward_vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=v_optimiser,
    )
    cost_qf_state = RLTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": qf_key}, obs, action),
        target_params=qf.init({"params": qf_key}, obs, action),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    cost_vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(actor_key, obs),
        tx=optax.adamw(learning_rate=args.policy_lr),
    )

    # Define update functions here to limit the need for static argname
    def get_V(actor_state, params, observations, subkey):
        mean, log_std = actor.apply(actor_state.params, observations)
        expert_next_state_actions, log_prob = sample_action_and_log_prob(
            mean, log_std, subkey
        )

        qf_next_target_values = qf.apply(
            params, observations, expert_next_state_actions
        )
        return qf_next_target_values - jnp.exp(0.001) * log_prob.reshape(-1, 1)

    checkpointer = orbax.checkpoint.PyTreeCheckpointer()

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    try:
        _, (obs, info) = envs.reset()
    except ValueError:  # Catch the exception if envs.reset() doesn't return a tuple
        obs, info = envs.reset()
    obs = obs.astype(np.float32)
    # Display progress bar if available
    cum_reward = 0
    cum_cost = 0
    lengths = 0
    agentstate = AgentState(
        actor_state,
        vf_state,
        qf_state,
        reward_qf_state,
        reward_vf_state,
        key,
    )
    dataset = env_module.offline_dataset()

    def normalize_datas(data, means, stds):
        return (data - means) / stds

    obs_means = (
        jnp.concatenate([expert_obs, dataset["observations"]])
        .astype(jnp.float32)
        .mean(0)
    )
    obs_stds = (
        jnp.concatenate([expert_obs, dataset["observations"]])
        .astype(jnp.float32)
        .std(0)
    )
    obs_stds = obs_stds.at[obs_stds < 1e-4].set(1)
    init_eff = 1 / (
        ((dataset["index"] == 0)).mean() * (1 - args.beta / 2)
        + (expert_timestep == 0).mean() * args.beta / 2
    )
    expert_obs = normalize_datas(expert_obs, obs_means, obs_stds)
    expert_next_obs = normalize_datas(expert_next_obs, obs_means, obs_stds)
    other_expert_obs = normalize_datas(other_expert_obs, obs_means, obs_stds)
    other_expert_next_obs = normalize_datas(other_expert_next_obs, obs_means, obs_stds)
    policy_obs = normalize_datas(
        dataset["observations"][..., : obs.shape[-1]].astype(jnp.float32),
        obs_means,
        obs_stds,
    )
    policy_next_obs = dataset["next_observations"] = normalize_datas(
        dataset["next_observations"][..., : obs.shape[-1]].astype(jnp.float32),
        obs_means,
        obs_stds,
    )
    policy_actions = jnp.array(dataset["actions"].astype(jnp.float32))
    policy_rewards = jnp.array(dataset["rewards"].astype(jnp.float32))
    policy_dones = jnp.array(
        (dataset["terminals"] * (1 - dataset["timeouts"])).astype(jnp.bool_)
    )
    policy_timestep = jnp.array(dataset["index"])

    policy_obs = jnp.concatenate([policy_obs, other_expert_obs])
    policy_next_obs = jnp.concatenate([policy_next_obs, other_expert_next_obs])
    policy_actions = jnp.concatenate([policy_actions, other_expert_actions])
    policy_rewards = jnp.concatenate([policy_rewards, other_expert_rewards])
    policy_dones = jnp.concatenate([policy_dones, other_expert_dones])
    policy_timestep = jnp.concatenate([policy_timestep, other_expert_timestep])

    @flax.struct.dataclass
    class WeightBuffer:
        weights: np.ndarray
        count: int

    weight_buffer = WeightBuffer(
        jnp.empty((total_step // update_preiod, len(policy_obs))), 0
    )

    @partial(jax.jit, donate_argnums=(0,))
    def update_weight_buffer(weight_buffer, reward_vf_state, vf_state):
        obs = policy_obs
        next_obs = policy_next_obs
        rewards = policy_rewards
        next_cost = vf.apply(vf_state.params, next_obs * single_mask).reshape(-1)
        tdv = (
            args.gamma * vf.apply(reward_vf_state.params, next_obs).reshape(-1)
            + rewards
            - vf.apply(reward_vf_state.params, obs).reshape(-1)
            + 2 * args.alpha
        )
        return (
            WeightBuffer(
                weight_buffer.weights.at[weight_buffer.count].set(tdv),
                weight_buffer.count + 1,
            ),
            tdv,
        )

    @jax.jit
    def get_positive_rate(reward_vf_state):
        obs = policy_obs
        next_obs = policy_next_obs
        rewards = policy_rewards
        y1 = (
            args.gamma * vf.apply(reward_vf_state.params, next_obs).reshape(-1)
            + rewards
            - vf.apply(reward_vf_state.params, obs).reshape(-1)
        )
        obs = expert_obs
        next_obs = expert_next_obs
        rewards = expert_rewards
        y2 = (
            args.gamma * vf.apply(reward_vf_state.params, next_obs).reshape(-1)
            + rewards
            - vf.apply(reward_vf_state.params, obs).reshape(-1)
        )
        return (omega_star(y1 / args.alpha) > 0).mean(), (
            omega_star(y2 / args.alpha) > 0
        ).mean()

    _train_step = jax.jit(train_step)
    _reward_train_step = jax.jit(reward_train_step)

    def train_step_body(carry, step):
        agentstate, weight_buffer = carry
        agentstate, callback_log = _train_step(agentstate, weight_buffer)
        return (agentstate, weight_buffer), callback_log

    def reward_train_step_body(carry, step):
        agentstate = carry
        agentstate, callback_log = _reward_train_step(agentstate)
        return agentstate, callback_log

    for global_step in range(total_step):
        if global_step != 0 and global_step % update_preiod == 0:
            eval_policy(envs, agentstate, global_step * step_per_scan)
            weight_buffer, update_weights = update_weight_buffer(
                weight_buffer, agentstate.reward_vf_state, agentstate.vf_state
            )
            (agentstate, _), callback_log = jax.lax.scan(
                train_step_body,
                (agentstate, weight_buffer),
                (),
                length=step_per_scan,
            )
            stats_log_data = jax.tree.map(jnp.mean, callback_log)
            stats_log_data = stats_log_data | {
                "charts/update_positive": (update_weights > 0).mean(),
            }
            state = {
                "qf_state": agentstate.qf_state,
                "vf_state": agentstate.vf_state,
                "reward_qf_state": agentstate.reward_qf_state,
                "reward_vf_state": agentstate.reward_vf_state,
                "actor_state": agentstate.actor_state,
            }
            actor_key, qf_key = jax.random.split(agentstate.key)
            agentstate = agentstate._replace(
                reward_vf_state=TrainState.create(
                    apply_fn=vf.apply,
                    params=vf.init(actor_key, obs),
                    tx=v_optimiser,
                ),
                actor_state=TrainState.create(
                    apply_fn=actor.apply,
                    params=actor.init(actor_key, obs),
                    tx=actor_optimiser,
                ),
            )
        else:
            stats_log_data = {}
        agentstate, reward_callback_log = jax.lax.scan(
            reward_train_step_body,
            (agentstate),
            (),
            length=step_per_scan,
        )
        stats_log_data = stats_log_data | jax.tree.map(jnp.mean, reward_callback_log)
        (policy_cover_rate, expert_cover_rate) = get_positive_rate(
            agentstate.reward_vf_state
        )

        log_data = {
            "charts/SPS": int(
                (global_step * step_per_scan) / (time.time() - start_time)
            ),
            "charts/expert_log_prob": get_log_prob(
                agentstate.actor_state.params, expert_actions, expert_obs
            ),
            "charts/policy_cover_rate": policy_cover_rate,
            "charts/expert_cover_rate": expert_cover_rate,
        }

        for k, v in log_data.items():
            print(k, v, global_step)
        if args.track:
            wandb.log(
                {"global_step": (global_step + 1) * step_per_scan, **stats_log_data}
            )
            wandb.log({"global_step": (global_step + 1) * step_per_scan, **log_data})

    eval_policy(envs, agentstate, total_step * step_per_scan, 100)
    state = {
        "qf_state": agentstate.qf_state,
        "vf_state": agentstate.vf_state,
        "reward_qf_state": agentstate.reward_qf_state,
        "reward_vf_state": agentstate.reward_vf_state,
        "actor_state": agentstate.actor_state,
    }
    save_args = orbax_utils.save_args_from_target(state)
    checkpointer.save(f"sacpolicy/{args.env_id}/{run_name}", state, save_args=save_args)
    # envs.close()
