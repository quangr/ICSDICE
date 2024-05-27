import numpy as np
import gymnasium as gym
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, ImageMagickWriter

class Plot2D:
    """
    Contains 2D plotting functions (done through matplotlib)
    """

    def __init__(self, l={}, cmds=[], mode="static", interval=None,
                 n=None, rows=1, cols=1, gif=None, legend=False, **kwargs):
        """
        Initialize a Plot2D.
        """
        assert (mode in ["static", "dynamic"])
        assert (type(l) == dict)
        assert (type(cmds) == list)
        self.legend = legend
        self.gif = gif
        self.fps = 60
        self.l = l
        self.mode = mode
        if interval is None:
            self.interval = 200  # 200ms
        else:
            self.interval = interval
        self.fig, self.ax = plt.subplots(nrows=rows, ncols=cols, **kwargs)
        self.empty = True
        self.cmds = cmds
        self.n = n
        self.reset()
        if self.mode == "dynamic":
            if n is None:
                self.anim = FuncAnimation(self.fig, self.step,
                                          blit=False, interval=self.interval, repeat=False)
            else:
                self.anim = FuncAnimation(self.fig, self.step,
                                          blit=False, interval=self.interval, frames=range(n + 1),
                                          repeat=False)

    def reset(self):
        """
        Reset and draw initial plots.
        """
        self.t = 0
        self.clear()
        self.data = {}
        self.objs = {}
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=None, t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=None, t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def step(self, frame=None):
        """
        Increment the timer.
        """
        self.t += 1
        for key, val in self.l.items():
            self.data[key] = val(p=self, l=self.data, t=self.t)
        for i, cmd in enumerate(self.cmds):
            if type(cmd) == list:
                if cmd[0](p=self, l=self.data, t=self.t):
                    self.objs[i] = cmd[1](p=self, l=self.data, o=self.objs[i], t=self.t)
            else:
                self.objs[i] = cmd(p=self, l=self.data, o=self.objs[i], t=self.t)
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        if self.legend:
                            item2.legend(loc='best')
                        item2.relim()
                        item2.autoscale_view()
                else:
                    if self.legend:
                        item.legend(loc='best')
                    item.relim()
                    item.autoscale_view()
        else:
            if self.legend:
                self.ax.legend(loc='best')
            self.ax.relim()
            self.ax.autoscale_view()

    def getax(self, loc=None):
        """
        Get the relevant axes object.
        """
        if loc is None:
            axobj = self.ax
        elif type(loc) == int or (type(loc) == list and len(loc) == 1):
            loc = int(loc)
            axobj = self.ax[loc]
        else:
            assert (len(loc) == 2)
            axobj = self.ax[loc[0], loc[1]]
        return axobj

    def imshow(self, X, loc=None, o=None, **kwargs):
        """
        Imshow X.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            im = axobj.imshow(X, **kwargs)
            cbar = self.fig.colorbar(im, ax=axobj)
            return [im, cbar]
        else:
            im, cbar = o
            im.set_data(X)
            im.set_clim(np.min(X), np.max(X))
            # cbar.set_clim(np.min(X), np.max(X))
            # matplotlib.cm.ScalarMappable.set_clim
            return [im, cbar]

    def line(self, X, Y, loc=None, o=None, **kwargs):
        """
        Line plot X/Y where `loc` are the subplot indices.
        """
        self.empty = False
        axobj = self.getax(loc=loc)
        if o is None:
            return axobj.plot(X, Y, **kwargs)[0]
        else:
            o.set_data(X, Y)
            return o

    def line_binary(self, X, Y, loc=None, o=None, trends=None,
                    trend_colors=['grey', 'pink'], **kwargs):
        """
        Line with two colors.
        """
        assert (trends is not None)
        self.empty = False
        axobj = self.getax(loc=loc)
        ret = []
        n = len(X)
        lw = 3
        if n == 0:
            return None
        if o is not None and len(o) > 0:
            for oo in o:
                oo.remove()
        for i in range(n - 1):
            ret += [axobj.plot(X[i:i + 2], Y[i:i + 2], color=trend_colors[0] \
                if trends[i] == "-" else trend_colors[1], linewidth=lw, **kwargs)[0]]
        return ret

    def show(self, *args, **kwargs):
        """
        Show the entire plot in a nonblocking way.
        """
        if not self.empty:
            if not plt.get_fignums():
                # print("Figure closed!")
                return
            if hasattr(self, "shown") and self.shown is True:
                plt.draw()
                plt.pause(0.001)
                return
            if self.gif is None:
                plt.show(*args, **kwargs)
                self.shown = True
            else:
                assert (self.n is not None)
                plt.show(*args, **kwargs)
                self.shown = True
                self.anim.save(self.gif, writer=ImageMagickWriter(fps=self.fps,
                                                                  extra_args=['-loop', '1']),
                               progress_callback=lambda i, n: print("%d/%d" % (i, n)))

    def clear(self):
        """
        Clear the figure.
        """
        if type(self.ax) == np.ndarray:
            for item in self.ax:
                if type(item) == np.ndarray:
                    for item2 in item:
                        item2.cla()
                else:
                    item.cla()
        else:
            self.ax.cla()
        self.empty = True

class WallGridworld(gym.Env):
    """
    nxm Gridworld. Discrete states and actions (up/down/left/right/stay).
    Agent starts randomly.
    Goal is to reach the reward.
    Inspired from following work:
    github.com/yrlu/irl-imitation/blob/master/mdp/gridworld.py
    """

    def reset_model(self):
        pass

    def __init__(self, map_height, map_width, reward_states, terminal_states, n_actions,
                 visualization_path='./',
                 transition_prob=1.,
                 unsafe_states=[],
                 start_states=None):
        """
        Construct the environment.
        Reward matrix is a 2D numpy matrix or list of lists.
        Terminal cells is a list/set of (i, j) values.
        Transition probability is the probability to execute an action and
        end up in the right next cell.
        """
        # super(WallGridworld).__init__(model_path, frame_skip)
        self.h = map_height
        self.w = map_width
        self.reward_mat = np.zeros((self.h, self.w))
        for reward_pos in reward_states:
            self.reward_mat[reward_pos[0], reward_pos[1]] = 1
        for unsafe_pos in unsafe_states:
            self.reward_mat[unsafe_pos[0], unsafe_pos[1]] = -1
        assert (len(self.reward_mat.shape) == 2)
        # self.h, self.w = len(self.reward_mat), len(self.reward_mat[0])
        self.n = self.h * self.w
        self.terminals = terminal_states
        if n_actions == 9:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            self.action_space = gym.spaces.Discrete(9)
        elif n_actions == 8:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]  # effect of each movement
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            self.action_space = gym.spaces.Discrete(8)
        elif n_actions == 4:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # effect of each movement
            self.actions = [0, 1, 2, 3]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u'}
            self.action_space = gym.spaces.Discrete(4)
        else:
            raise EnvironmentError("Unknown number of actions {0}.".format(n_actions))
        self.transition_prob = transition_prob
        self.terminated = True
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]),
                                                high=np.array([self.h, self.w]), dtype=np.int32)
        self.unsafe_states = unsafe_states
        self.start_states = start_states
        self.steps = 0
        self.visualization_path = visualization_path
        self.terminal_step=50

    def get_states(self):
        """
        Returns list of all states.
        """
        return filter(
            lambda x: self.reward_mat[x[0]][x[1]] not in [-np.inf, float('inf'), np.nan, float('nan')],
            [(i, j) for i in range(self.h) for j in range(self.w)]
        )

    def get_actions(self, state):
        """
        Returns list of actions that can be taken from the given state.
        """
        if self.reward_mat[state[0]][state[1]] in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
            return [4]
        actions = []
        for i in range(len(self.actions) - 1):
            inc = self.neighbors[i]
            a = self.actions[i]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and 0 <= nei_s[1] < self.w and \
                    self.reward_mat[nei_s[0]][nei_s[1]] not in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                actions.append(a)
        return actions

    def terminal(self, state):
        """
        Check if the state is terminal.
        """
        if self.steps>self.terminal_step:
            return True
        for terminal_state in self.terminals:
            if state[0] == terminal_state[0] and state[1] == terminal_state[1]:
                return True
        return False

    def get_next_states_and_probs(self, state, action):
        """
        Given a state and action, return list of (next_state, probability) pairs.
        """
        if self.terminal(state):
            return [((state[0], state[1]), 1)]
        if self.transition_prob == 1:
            inc = self.neighbors[action]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if 0 <= nei_s[0] < self.h and \
                    0 <= nei_s[1] < self.w and \
                    self.reward_mat[nei_s[0]][nei_s[1]] not in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                return [(nei_s, 1)]
            else:
                return [((state[0], state[1]), 1)]  # state invalid
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs[action] = self.transition_prob
            mov_probs += (1 - self.transition_prob) / self.n_actions
            for a in range(self.n_actions):
                inc = self.neighbors[a]
                nei_s = (state[0] + inc[0], state[1] + inc[1])
                if nei_s[0] < 0 or nei_s[0] >= self.h or \
                        nei_s[1] < 0 or nei_s[1] >= self.w or \
                        self.reward_mat[nei_s[0]][nei_s[1]] in \
                        [-np.inf, float('inf'), np.nan, float('nan')]:
                    # mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0
            # sample_action = random.choices([i for i in range(self.n_actions)], weights=mov_probs, k=1)[0]
            # inc = self.neighbors[sample_action]
            # return [((state[0] + inc[0], state[1] + inc[1]), 1)]
            res = []
            mov_probs = mov_probs * 1/np.sum(mov_probs)
            for a in range(self.n_actions):
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    nei_s = (state[0] + inc[0], state[1] + inc[1])
                    res.append((nei_s, mov_probs[a]))
            return res

    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state

    def pos2idx(self, pos):
        """
        Convert column-major 2d position to 1d index.
        """
        return pos[0] + pos[1] * self.h

    def idx2pos(self, idx):
        """
        Convert 1d index to 2d column-major position.
        """
        return (idx % self.h, idx // self.h)

    def reset_with_values(self, info_dict):
        self.curr_state = info_dict['states']
        assert self.curr_state not in self.terminals
        self.terminated = False
        self.steps = 0
        return self.state

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        if 'states' in kwargs.keys():
            self.curr_state = kwargs['states']
            assert self.curr_state not in self.terminals
            self.terminated = False
            self.steps = 0
            return self.state
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
            return self.state,{}

    def step(self, action):
        """
        Step the environment.
        """
        action = int(action)
        st_prob = self.get_next_states_and_probs(self.state, action)
        sampled_idx = np.random.choice(np.arange(0, len(st_prob)), p=[prob for st, prob in st_prob])
        next_state = st_prob[sampled_idx][0]
        reward = self.reward_mat[next_state[0]][next_state[1]]
        self.curr_state = next_state
        self.steps += 1
        admissible_actions = self.get_actions(self.curr_state)
        if self.terminal(self.state):
            self.terminated = True
            return (list(self.state),
                    reward,
                    True,
                    True,
                    {'x_position': self.state[0],
                     'y_position': self.state[1],
                     'admissible_actions': admissible_actions,
                     },
                    )
        self.terminated = False
        return (list(self.state),
                reward,
                False,
                False,
                {'y_position': self.state[0],
                 'x_position': self.state[1],
                 'admissible_actions': admissible_actions,
                 },
                )

    def seed(self, s=None):
        """
        Seed this environment.
        """
        random.seed(s)
        np.random.seed(s)

    def render(self, mode, **kwargs):
        """
        Render the environment.
        """
        self.state_mat = np.zeros([self.h, self.w, 3])
        self.state_mat[self.state[0], self.state[1], :] = [255, 255, 255]
        self.state_mat += self.reward_mat[...,None]*[0, 255, 0]
        for state in self.unsafe_states:
            self.state_mat[state[0], state[1], :] = [255, 0, 0]

        plt.imshow(self.state_mat)
        plt.draw()
        plt.pause(1/2)
        # if not hasattr(self, "plot"):
        #     self.plot = Plot2D({
        #         "env": lambda p, l, t: self,
        #     }, [
        #         [
        #             lambda p, l, t: not l["env"].terminated,
        #             lambda p, l, o, t: p.imshow(l["env"].state_mat, o=o)
        #         ],
        #     ], mode="dynamic", interval=1)
        # self.plot.show(block=False)

        # # if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
        # if mode == "rgb_array":
        #     self.plot.fig.canvas.draw()
        #     img = np.frombuffer(self.plot.fig.canvas.tostring_rgb(), dtype=np.uint8)
        #     img = img.reshape(self.plot.fig.canvas.get_width_height()[::-1] + (3,))
        #     plt.pause(1/2)
        #     return img
