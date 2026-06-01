# Balance Runs with Unitree's H1 humanoid

## Run 001

### Hypothesis:

If I build out the code, I can make a humanoid do something.

### Changes:

- Built the code.
- The initial structure is about as sparse of a reward landscape as they come: 1 single reward, and it is a penalty for falling over.

### Result:

~~Nothing! Partly expected. No learning happened in the 500 iterations, but it ran! Not the humanoid, but the code. Maybe one day the humanoid will run. That'd be cool.~~

Spoke too soon. Although some of the tensorboard metrics showed no progress being made (specifically mean_reward staying flat), the mean_episode_length actually increased throughout the 500 iterations. After watching the checkpoints, I noticed there is learning happening. By iteration 500, the H1s learned to shimmy their leg joints, likely to create instability that delayed their falls.

## Run 002

### Hypothesis:

The sparse reward (1 penalty for falling over) actually showed a glimpse of learning throughout 500 iterations. It would be wise to extend the training, just to see how far this single reward can push it.

### Changes:

- Increased iterations from 500 to 3000

### Result:

Interesting result. By iteration 600, the H1s developed clearly superior balancing behavior compared to just falling over. They began to start with a more wide-legged stance and a sideways torso, using their torso to help balance and (usually starting with the right leg) shifting their legs one after the other to maintain balance. The closer the H1s torso got to the ground, the more shimmy and shifting occurred. By iteration 1500, the H1s effectively learned the spazzing technique, where they quickly alternated spazzing their lower leg joints up and down to maintain a very odd-looking balance. This proved effective, as episode termination by falling over (torso contacting the ground) dropped down to 30%, and as low as 16% by iteration 2700. Most of the iterations (~1000-2800) were spent perfecting this spazzing technique. An interesting decline happened toward the end of training, around iteration ~2900, where the spazzing got out of hand. The spazzing grew too violent, occasionally lifting the H1s doing it off the ground and floating them in the air long enough to tilt the humanoid, leading to a greater likelihood of the H1 losing balance and contacting the ground. So, there does seem to be a limit to this erratic, spazzing technique.

## Run 003

### Hypothesis:

While it is impressive that the H1s learned to somewhat balance (chaotically), the single penalty is far too sparse and not descriptive enough for the agent to learn human-like, stable balancing behavior. By adding a small ankle deviation penalty (small penalties for non-default ankle positioning), the H1s have another reward signal to learn from. Human ankles play a big role in balance by correcting many small disturbances before they become large enough to affect the hips and core. Just this small penalty alone should push the H1s to learn a more stable, less chaotic, less spazzy balancing behavior.

### Changes:

- Added ankle deviation penalty with weight -0.05

### Result:

The general learned behavior changed drastically from run 002. The H1s learned how to minimize the ankle penalty from i1 to i200, but episode_length declined due to the H1s' legs locking up in front of them. The H1s did bring the episode_length back up to roughly where it started by widening its leg stance, but they still fell over due to the ankles locking up. The one major difference between this run and the last is that the H1s tend to fall over sideways on this run, whereas they usually fell over forwards or backwards in the last.
