## Runs 001-003:

The biggest thing I have learned so far is that the reward landscape for humanoid locomotion is largely penalty-shaped. In my previous Unity project, training tanks to capture points (game AI), I learned the difference between pessimistic and optimistic reward landscapes. I'd shifted my rewards from pessimistic (penalty-heavy) to optimistic (reward-heavy), but this does not carry over well for humanoid locomotion.

Humanoid locomotion is penalty-shaped due to the desired behavior(s). Whereas my virtual tank AIs could accomplish their tasks in a different set of ways, humanoid locomotion (stability in particular) has a large window of failures (jerking motions, running very close to the ground, jumping incredibly high each step) for a very narrow intended behavior (balancing in place, walking with a human-like gait, jogging similarly).

Another difference from ML-Agents that I have had to get used to is the tensorboard metrics. The most parallel envs I had running during any of my tank agent runs was 4. For this project, it is normal to run 1024+. The tensorboard metrics for IsaacLab are per-step averages, whereas Unity's ML-Agents metrics were average rewards in an episode.

The maximum reward of a 0.05 weight is_alive in an episode with 1000 steps is 50 (1000 x 0.05), but tensorboard will show me 0.04 (per-step average). This normalized, per-step metric helps metrics stay the same, whether or not I start training with 2048 or 4096 envs in future runs.

## Runs 004-006:

I have started learning about L1 and L2 norms, and why mdp rewards use one or the other. L2 squared is primarily used in most of the rewards(penalties) due to the fact that it can heavily penalize large errors while still tolerating small errors. This allows the policy to learn gradually (PPO).

I've learned that humanoids differ from quadrupeds in their natural lack of balance. A robodog can balance far more easily than a two-legged humanoid.

I've also learned that it may be wiser to start humanoid tasks with a relatively denser reward landscape, as the humanoids will likely have more refined behaviors to go off of. The H1s in my first run (just a termination penalty) did learn a little bit, but it was hard to figure out what rewards I may need to add due to the spastic, erratic behavior. Now, in run 5 specifically, there is a clearer behavior exhibited: slow balance that causes the H1s to drift forward. This is far easier to tweak/diagnose than the spastic H1s from run 1.

## Run 006:

I've learned that there is a difference in the policy that I watch in checkpoint replays versus actual training. I saw that i999 of run 006 had episode terminations by torso contact (falling over) at ~65%, but watching .pt from i999 showed the 32 robots successfully balancing 100% of the time. This is due to how training works. PPO deliberately adds noise to the training so it can learn (making it a stochastic policy), whereas watching a .pt file shows a deterministic policy (no noise). This is due to rsl_rl's built-in functionality of switching to inference mode when running play scripts.

I have also learned that disabling corruption (self.observations.policy.enable_corruption = False) in the observation space isn't the best practice. Even though it will be a long while before I get to engage in sim-to-real transfer (anybody got a spare robot to lend me?), it is best to introduce observation corruption to my play script to mimic real hardware. The training script has corruption enabled, so enabling it on my play script will show more accurately what the policy actually looks like. It may be useful to switch on-and-off for runs that are interesting enough.

## Runs 006-009:

I've learned that N.m stands for Newton-Meters, a form of torque measurement. I've also learned that the H1, like many other humanoids presumably, has motors with differing capacities of torque available. The knee joint has the largest potential amount of torque in a joint, sitting at 360 N.m max, whereas other joints have less, like the ankle joint at 45 N.m max. This plays a key factor in reward design decisions, as penalizing all joint torques equally with a penalty like joint_torques_l2 disproportionately affects joints like the ankle while underpenalizing joints like the knee.
