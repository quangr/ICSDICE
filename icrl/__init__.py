from icrl.env.wrapper.mazewrapper import ConcatenateObservationNoGoal
from gymnasium.envs.registration import register
register(
    id="Maze2d_simple",
    entry_point="gymnasium_robotics.envs.maze.point_maze:PointMazeEnv",
    kwargs={
        "continuing_task": False,
        "reward_type": "sparse",
        "maze_map": [
            [1, 1, 1, 1, 1],
            [1, "g", 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, "r", 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
    },
    additional_wrappers=(ConcatenateObservationNoGoal.wrapper_spec(),),
    max_episode_steps=300,
)
try:
   import commonroad_rl.gym_commonroad
except ImportError:
   pass