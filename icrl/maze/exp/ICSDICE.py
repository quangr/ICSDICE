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


AgentState = collections.namedtuple(
    "AgentState",
    [
        "reward_vf_state",
        "key",
    ],
)


vf = ValueCritic()


@dataclass
class ICSDICE:
    gamma: float = 0.99
    alpha: float = 0.1
    UseIS: bool = True
    UseDiscount: bool = True
    init_eff: float = 1

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
                if self.UseIS:
                    init_loss = (
                        (1 - gamma)
                        * self.init_eff
                        * ((Batch.index == 0) * current_V).mean()
                    )
                else:
                    init_loss = (1 - gamma) * (init_V).mean()
                if self.UseDiscount:
                    TotalN = next_feasible.sum() + Batch.dones.sum()
                    y = gamma * next_V - current_V + Batch.rewards
                    loss = (
                        init_loss
                        + alpha
                        * (
                            next_feasible * (gamma**Batch.index) * fp_star(y / alpha)
                        ).sum()
                        / TotalN
                    ) + alpha * (
                        Batch.dones
                        * (gamma ** (Batch.index + 1))
                        / (1 - gamma)
                        * fp_star((gamma * next_V - next_V) / alpha)
                    ).sum() / TotalN
                else:
                    TotalN = next_feasible.sum()
                    y = gamma * (1 - Batch.dones) * next_V - current_V + Batch.rewards
                    loss = (
                        init_loss
                        + alpha * (next_feasible * fp_star(y / alpha)).sum() / TotalN
                    )
                return loss

            rewad_vf_loss_value, grads = jax.value_and_grad(reward_loss)(
                reward_vf_state.params
            )
            callback_log["rewad_vf_loss_value"] = rewad_vf_loss_value
            reward_vf_state = reward_vf_state.apply_gradients(grads=grads)
            return reward_vf_state, key

        (
            reward_vf_state,
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
                key,
            ),
            callback_log,
        )

    def get_tdv(self, agent_state, obs, next_obs, rewards):
        next_v = vf.apply(agent_state.reward_vf_state.params, next_obs).reshape(-1)
        v = vf.apply(agent_state.reward_vf_state.params, obs).reshape(-1)
        tdv = (self.gamma * next_v + rewards - v) / (2 * self.alpha)
        return tdv, v


@dataclass
class ICSDICENoIS(ICSDICE):
    UseIS: bool = False


@dataclass
class ICSDICENoDiscount(ICSDICE):
    UseDiscount: bool = False


@dataclass
class ICSDICEBaseline(ICSDICE):
    UseIS: bool = False
    UseDiscount: bool = False
