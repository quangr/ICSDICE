# Learning Constraints from Offline Demonstrations via Superior Distribution Correction Estimation

This is the code for the paper "Learning Constraints from Offline Demonstrations via Superior Distribution Correction Estimation" published at ICML 2024. The implementation is based on the code from [Cleanrl](https://github.com/vwxyzjn/cleanrl/) and [jax-rl](https://github.com/quangr/jax-rl).

# Setup Experimental Environments

You may run the following command to install dependencies:

```
pip install -r requirements.txt
pip install -U "jax[cuda]==0.4.23" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install git+https://github.com/quangr/dejax.git
```

# Point Maze 

to run Point Maze compare: `jupyter nbconvert --to notebook --execute --inplace icrl/maze/exp/train.ipynb --ExecutePreprocessor.timeout=-1`

# Grid-World

To run Grid-World env with our method:

 ` jupyter nbconvert --to notebook --execute --inplace icrl/grid_world/ICSDICE.ipynb --ExecutePreprocessor.timeout=-1`

To run Grid-World env with RECOIL-V:

 `python icrl/grid_world/recoil-V.py`

# Mujoco

To run mujoco env with our method:

`python icrl/benchmark/ICSDICE.py --seed 1 --alpha=0.0001 --total-timesteps=1000000 --debug=False --update-period=100000 --beta=0.5 --env-id=Ant_ls --cost-l2=0.0005 --cost-limit=0.9 --expert-ratio 1.0`
