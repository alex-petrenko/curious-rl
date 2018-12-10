# curious-rl

Re-implementation of the 2017 paper "Curiosity-driven Exploration by Self-supervised Prediction" by Deepak Pathak et al.
[arxiv](https://arxiv.org/pdf/1705.05363.pdf)

ICM module is added to the A2C algorithm, instead of A3C in the original paper. Includes sparse rewards Doom mazes from
the original paper.

Curious agent solves environment "VizdoomMyWayHome-v0":

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_maze.gif?raw=true" width="400" >
<br />

In under 100M frames the curious agent was able to solve "VizdoomMyWayHomeVerySparse-v0", which vanilla A2C
fails to solve even after training on 500M frames.

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_very_sparse.gif?raw=true" width="400" >

<br />

If the hyperparameters are not right the curious agent can get stuck in funny local minima, e.g. in this case the
learning rate of the predictive model was too slow, and the agent was forever curious about this wall with the
bright texture. 

<img src="https://github.com/alex-petrenko/curious-rl/blob/master/files/gifs/doom_failure.gif?raw=true" width="400" >
