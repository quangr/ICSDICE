import gymnasium as gym
import numpy as np
import jax.numpy as jnp
import pygame
import random
# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
env_id="WallGridworld-v0"
def cost_function(next_obs, reward, next_done, next_truncated, info):
    return jnp.zeros_like(reward)
class WallGridworldRender:
    def __init__(self, env):
        self.env = env
        self.cell_size = 50
        self.width = self.env.w * self.cell_size
        self.height = self.env.h * self.cell_size

        self.window = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("WallGridworld")

    def draw_grid(self):
        for y in range(self.env.h):
            for x in range(self.env.w):
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, WHITE, rect, 1)

    def draw_walls(self):
        for y in range(self.env.h):
            for x in range(self.env.w):
                if self.env.reward_mat[y, x] == -1:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.window, BLACK, rect)

    def draw_player(self, position):
        y, x = position
        rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
        pygame.draw.rect(self.window, GREEN, rect)

    def draw_terminal_states(self):
        for terminal_state in self.env.terminals:
            y, x = terminal_state
            rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
            pygame.draw.rect(self.window, RED, rect)

    def render(self):
        self.window.fill(BLACK)
        self.draw_grid()
        self.draw_walls()
        self.draw_terminal_states()
        self.draw_player(self.env.state)

        pygame.display.flip()

    def run(self):
        states = []
        next_states = []
        dones = []
        actions = []

        done = False
        state, _ = self.env.reset()
        self.draw_player(state)
        self.render()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        action = 1
                    elif event.key == pygame.K_RIGHT:
                        action = 0
                    elif event.key == pygame.K_UP:
                        action = 3
                    elif event.key == pygame.K_DOWN:
                        action = 2
                    else:
                        action = None
                    
                    if action is not None:
                        actions.append(action)
                        states.append(state)
                        state, reward, done, _, _ = self.env.step(action)
                        print(state,action, reward, done)
                        next_states.append(state)
                        dones.append(done)
                        self.draw_player(state)
                        self.render()
        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        dones = np.array(dones)
        print("expert_action = np.array(" + np.array2string(actions, separator=', ') + ")")
        print("expert_obs = np.array(" + np.array2string(states, separator=', ') + ")")
        print("expert_next_obs = np.array(" + np.array2string(next_states, separator=', ') + ")")
        print("expert_dones = np.array(" + np.array2string(dones, separator=', ') + ")")
        pygame.quit()

class WallGridworld(gym.Env):
    def __init__(self, map_height, map_width, reward_states, terminal_states, n_actions,
                 transition_prob=1., unsafe_states=[], start_states=None):
        self.h = map_height
        self.w = map_width
        self.reward_mat = np.zeros((self.h, self.w))
        for reward_pos in reward_states:
            self.reward_mat[reward_pos[0], reward_pos[1]] = 1
        # for unsafe_pos in unsafe_states:
        #     self.reward_mat[unsafe_pos[0], unsafe_pos[1]] = -1
        assert (len(self.reward_mat.shape) == 2)

        self.n = self.h * self.w
        self.terminals = terminal_states

        actions_dict = {
            9: [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)],
            8: [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)],
            4: [(0, 1), (0, -1), (1, 0), (-1, 0)],
        }

        if n_actions not in actions_dict:
            raise EnvironmentError("Unknown number of actions {0}.".format(n_actions))

        self.neighbors = actions_dict[n_actions]
        self.actions = list(range(n_actions))
        self.n_actions = n_actions

        dirs_dict = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
        self.dirs = {i: dirs_dict[i] for i in self.actions}

        self.transition_prob = transition_prob
        self.terminated = True
        self.unsafe_states = unsafe_states
        self.start_states = start_states
        self.steps = 0
        self.terminal_step = 50


    @property
    def observation_space(self):
        return gym.spaces.Box(low=np.array([0]*(self.h+self.w)), high=np.array([1]*(self.h+self.w)), dtype=np.int32)

    @property
    def action_space(self):
        return gym.spaces.Discrete(self.n_actions)

    def get_states(self):
        return filter(lambda x: self.reward_mat[x[0]][x[1]] not in [-np.inf, float('inf'), np.nan, float('nan')],
                      [(i, j) for i in range(self.h) for j in range(self.w)])

    def get_actions(self, state):
        if self.reward_mat[state[0]][state[1]] in [-np.inf, float('inf'), np.nan, float('nan')]:
            return [4]
        actions = []
        for i, (inc, a) in enumerate(zip(self.neighbors, self.actions)):
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and 0 <= nei_s[1] < self.w and self.reward_mat[nei_s[0]][nei_s[1]] not in [-np.inf, float('inf'), np.nan, float('nan')]:
                actions.append(a)
        return actions

    def terminal(self, state):
        return self.steps > self.terminal_step or state in self.terminals

    def get_next_states_and_probs(self, state, action):
        if self.terminal(state):
            return [((state[0], state[1]), 1)]

        mov_probs = np.zeros([self.n_actions])
        mov_probs[action] = self.transition_prob
        mov_probs += (1 - self.transition_prob) / self.n_actions

        res = []
        for a in range(self.n_actions):
            inc = self.neighbors[a]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if nei_s[0] < 0 or nei_s[0] >= self.h or nei_s[1] < 0 or nei_s[1] >= self.w or self.reward_mat[nei_s[0]][nei_s[1]] in [-np.inf, float('inf'), np.nan, float('nan')]:
                mov_probs[a] = 0
        if not mov_probs.any():
            return [((state[0], state[1]), 1)]
        mov_probs = mov_probs * 1 / np.sum(mov_probs)
        for a in range(self.n_actions):
            if mov_probs[a] != 0:
                inc = self.neighbors[a]
                nei_s = (state[0] + inc[0], state[1] + inc[1])
                res.append((nei_s, mov_probs[a]))
        return res
    @property
    def state(self):
        return self.curr_state

    def pos2idx(self, pos):
        return pos[0] + pos[1] * self.h

    def idx2pos(self, idx):
        return (idx % self.h, idx // self.h)

    def reset_with_values(self, info_dict):
        self.curr_state = info_dict['states']
        assert self.curr_state not in self.terminals
        self.terminated = False
        self.steps = 0
        return self.state

    def state_to_onehot(self,state):
        one_hot = np.zeros((len(state), self.h + self.w))
        one_hot[state[:,0]>=0,  state[state[:,0]>=0,0]]=1
        one_hot[state[:,1]>=0, self.h + state[state[:,1]>=0,1]]=1
        return one_hot
    def tuple_to_onehot(self,state):
        one_hot = np.zeros(self.h + self.w)
        if state[0]>=0:
            one_hot[state[0]]=1
        if state[1]>=0:
            one_hot[self.h + state[1]]=1
        return one_hot


    def reset(self, **kwargs):
        if 'states' in kwargs.keys():
            self.curr_state = kwargs['states']
            assert self.curr_state not in self.terminals
        else:
            if self.start_states is not None:
                random_state = random.choice(self.start_states)
                self.curr_state = random_state
            else:
                random_state = np.random.randint(self.h * self.w)
                self.curr_state = self.idx2pos(random_state)
            while self.curr_state in self.terminals or self.curr_state in self.unsafe_states:
                if self.start_states is not None:
                    random_state = random.choice(self.start_states)
                    self.curr_state = random_state
                else:
                    random_state = np.random.randint(self.h * self.w)
                    self.curr_state = self.idx2pos(random_state)
        self.terminated = False
        self.steps = 0
        return self.tuple_to_onehot(self.state), {}

    def step(self, action):
        if self.terminated is True:
            return (self.tuple_to_onehot([-1,-1]),
                    self.reward_mat[self.state[0]][self.state[1]],
                    True,
                    self.steps == self.terminal_step,
                    {'x_position': self.state[0],
                     'y_position': self.state[1],
                     },
                    )

        action = int(action)
        forbidden=False
        st_prob = self.get_next_states_and_probs(self.state, action)
        if len(st_prob)==1 and st_prob[0][0]==self.curr_state:
            forbidden=True
        sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
        next_state = st_prob[sampled_idx][0]
        self.curr_state = next_state
        self.steps += 1
        admissible_actions = self.get_actions(self.curr_state)
        if self.terminal(self.state):
            self.terminated = True
        return (self.tuple_to_onehot(self.state),
                (-1 if forbidden else 0),
                forbidden,
                False,
                {'y_position': self.state[0],
                 'x_position': self.state[1],
                 'admissible_actions': admissible_actions,
                 },
                )

    def seed(self, s=None):
        np.random.seed(s)
        random.seed(s)


gym.register(
    id='WallGridworld-v0', 
    entry_point=WallGridworld, 
    reward_threshold=1.0, 
    nondeterministic=False, 
    max_episode_steps=50, 
    order_enforce=True, 
    autoreset=False, 
    disable_env_checker=False, 
    apply_api_compatibility=False, 
    additional_wrappers=(), 
    vector_entry_point=None, 
    kwargs={'map_height':5, 
    'map_width':5, 
    'reward_states':[(4, 0)], 
    'terminal_states':[(4, 0)], 
    'n_actions':4, 
    'transition_prob':1, 
    'unsafe_states':[], 
    'start_states':[(0,0)]}
)