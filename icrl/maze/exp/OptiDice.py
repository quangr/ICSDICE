from dataclasses import dataclass
import jax
import optax
import jax
from flax.training.train_state import TrainState
import collections
from icrl.maze.utility import (
    RLTrainState,
    ValueCritic,
    BatchData,
)


def omega_star(y):
    return (y > -2) * (y / 2 + 1)


def fp_star(y):
    return (y > -2) * (y * (y + 4) / 4)


vf = ValueCritic()
AgentState = collections.namedtuple(
    "AgentState",
    [
        "reward_vf_state",
        "lambda_state",
        "key",
    ],
)


@dataclass
class OptiDice:
    gamma: float = 0.99
    alpha: float = 0.1

    @staticmethod
    def GetTrainState(obs):
        key = jax.random.PRNGKey(0)
        key, actor_key, vf_key = jax.random.split(key, 3)
        lambda_state = TrainState.create(
            apply_fn=None, params=0.0, tx=optax.adamw(learning_rate=5e-3)
        )
        reward_vf_state = TrainState.create(
            apply_fn=vf.apply,
            params=vf.init(vf_key, obs),
            tx=optax.adam(learning_rate=5e-3),
        )
        agent_state = AgentState(
            reward_vf_state,
            lambda_state,
            key,
        )
        return agent_state

    def reward_train_step(self, agent_state, Batch):
        callback_log = {}
        alpha = self.alpha
        gamma = self.gamma

        def update_reward_value_critic(
            Batch: BatchData,
            reward_vf_state: RLTrainState,
            lambda_state: TrainState,
            key: jax.Array,
        ):
            next_feasible = 1 - Batch.costs

            def reward_loss(params):
                current_V = vf.apply(params, Batch.observations).reshape(-1)
                next_V = vf.apply(params, Batch.next_observations).reshape(-1)
                init_V = vf.apply(params, Batch.init_observations).reshape(-1)

                y = gamma * (1 - Batch.dones) * next_V - current_V + Batch.rewards
                lamb = lambda_state.params
                loss = (
                    (1 - gamma) * (init_V).mean()
                    + lamb
                    + alpha * (next_feasible * fp_star((y - lamb) / alpha)).mean()
                )
                return loss

            rewad_vf_loss_value, grads = jax.value_and_grad(reward_loss)(
                reward_vf_state.params
            )
            callback_log["rewad_vf_loss_value"] = rewad_vf_loss_value
            reward_vf_state = reward_vf_state.apply_gradients(grads=grads)

            def lambda_loss(lamb):
                params = reward_vf_state.params
                current_V = vf.apply(params, Batch.observations).reshape(-1)
                next_V = vf.apply(params, Batch.next_observations).reshape(-1)
                y = gamma * (1 - Batch.dones) * next_V - current_V + Batch.rewards
                loss = lamb + alpha * fp_star((y - lamb) / alpha).mean()

                callback_log["terminal_reg"] = jax.lax.stop_gradient(
                    (Batch.dones).sum()
                )
                return loss

            lambda_loss_value, grads = jax.value_and_grad(lambda_loss)(
                lambda_state.params
            )
            callback_log["lambda_vf_loss_value"] = lambda_loss_value
            lambda_state = lambda_state.apply_gradients(grads=grads)

            return reward_vf_state, lambda_state, key

        (
            reward_vf_state,
            lambda_state,
            key,
        ) = agent_state
        reward_vf_state, lambda_state, key = update_reward_value_critic(
            Batch,
            reward_vf_state,
            lambda_state,
            key,
        )
        return (
            AgentState(
                reward_vf_state,
                lambda_state,
                key,
            ),
            callback_log,
        )

    def get_tdv(self, agent_state, obs, next_obs, rewards):
        next_v = vf.apply(agent_state.reward_vf_state.params, next_obs).reshape(-1)
        v = vf.apply(agent_state.reward_vf_state.params, obs).reshape(-1)
        lamb = agent_state.lambda_state.params
        tdv = (self.gamma * next_v + rewards - v - lamb) / (2 * self.alpha)
        return tdv, v


@dataclass
class OptiDiceNoNormal:
    gamma: float = 0.99
    alpha: float = 0.1

    @staticmethod
    def GetTrainState(obs):
        key = jax.random.PRNGKey(0)
        key, actor_key, vf_key = jax.random.split(key, 3)
        reward_vf_state = TrainState.create(
            apply_fn=vf.apply,
            params=vf.init(vf_key, obs),
            tx=optax.adam(learning_rate=5e-3),
        )
        agent_state = AgentState(
            reward_vf_state,
            None,
            key,
        )
        return agent_state

    def reward_train_step(self, agent_state, Batch):
        callback_log = {}
        alpha = self.alpha
        gamma = self.gamma

        def update_reward_value_critic(
            Batch: BatchData,
            reward_vf_state: RLTrainState,
            key: jax.Array,
        ):
            next_feasible = 1 - Batch.costs

            def reward_loss(params):
                current_V = vf.apply(params, Batch.observations).reshape(-1)
                next_V = vf.apply(params, Batch.next_observations).reshape(-1)
                init_V = vf.apply(params, Batch.init_observations).reshape(-1)

                y = gamma * (1 - Batch.dones) * next_V - current_V + Batch.rewards
                loss = (1 - gamma) * (init_V).mean() + alpha * (
                    next_feasible * fp_star(y / alpha)
                ).mean()
                return loss

            rewad_vf_loss_value, grads = jax.value_and_grad(reward_loss)(
                reward_vf_state.params
            )
            callback_log["rewad_vf_loss_value"] = rewad_vf_loss_value
            reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
            return reward_vf_state, key

        (
            reward_vf_state,
            _,
            key,
        ) = agent_state
        reward_vf_state, key = update_reward_value_critic(
            Batch,
            reward_vf_state,
            key,
        )
        return (
            AgentState(
                reward_vf_state,
                None,
                key,
            ),
            callback_log,
        )

    def get_tdv(self, agent_state, obs, next_obs, rewards):
        next_v = vf.apply(agent_state.reward_vf_state.params, next_obs).reshape(-1)
        v = vf.apply(agent_state.reward_vf_state.params, obs).reshape(-1)
        tdv = (self.gamma * next_v + rewards - v) / (2 * self.alpha)
        return tdv, v
