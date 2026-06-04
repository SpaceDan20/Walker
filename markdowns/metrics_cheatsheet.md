Mean action noise std: PPO exploration. High --> Low as policy converges.
Mean value_function loss: How wrong critic value estimates are. Lower = Better.
Mean surrogate loss: PPO policy loss (how much the policy is changing each update). Small positive = healthy.
Mean entropy loss: How random the policy's action distribution is. Higher = More Random = Exploration.
Mean episode length: Average steps per episode.

When looking at metrics like surrogate and entropy loss, a small negative value (~-0.01) is the small healthy "positive".
