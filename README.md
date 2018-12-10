# Curious RL

Re-implementation of the 2017 paper "Curiosity-driven Exploration by Self-supervised Prediction" by Deepak Pathak et al.
([arXiv link](https://arxiv.org/pdf/1705.05363.pdf)). This was a project for the course EE556 in University of Southern California.

Original imlementation by the author can be found here: https://github.com/pathak22/noreward-rl

In this implementation the ICM module is added to the A2C algorithm, instead of A3C in the original paper.
Here curious agent solves environment "VizdoomMyWayHome-v0":

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_maze.gif?raw=true" width="320">

In under 100M frames the curious agent was able to solve "VizdoomMyWayHomeVerySparse-v0", which vanilla A2C
fails to solve even after training on 500M frames.

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_very_sparse.gif?raw=true"  width="320">

If the hyperparameters are not right the curious agent can get stuck in funny local optima, e.g. in this case the
learning rate of the predictive model was too low, and the agent was forever curious about this wall with the
bright texture. 

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_failure.gif?raw=true" width="320">

#### Installation

This repository uses `pipenv`, a tool that manages both virtualenvs and Python dependencies. Install
it if you don't have it:

```shell
pip install pipenv
```

clone the repo and create a virtualenv with all the packages, activate the env:

```shell
git clone https://github.com/alex-petrenko/curious-rl.git
cd curious-rl
pipenv sync
pipenv shell
```

run tests:

```shell
python -m unittest
```

#### Experiments

Train curious agents in different environments. Use tensorboard to monitor the training process, stop when necessary:

```shell
python -m algorithms.curious_a2c.train_curious_a2c --env=doom_basic
python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze
python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze_sparse
python -m algorithms.curious_a2c.train_curious_a2c --env=doom_maze_very_sparse

tensorboard --logdir ./train_dir
```

The latest model will be saved periodically. After training to desired performance you can examine agent's
behavior:

```shell
python -m algorithms.curious_a2c.enjoy_curious_a2c --env=doom_maze_sparse
```

It may take a long time to train the agent on mazes, be patient.
The detailed [PDF project report](https://github.com/alex-petrenko/curious-rl/blob/master/files/project_report.pdf) is available.

If you have any questions about this repo please feel free to reach me: apetrenko1991@gmail.com
