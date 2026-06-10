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

## Run 010:

I've learned that torques can be shown as either negative or positive, depending on how the motor orients the "positive" direction. So for a knee joint, a motor trying to extend the knee may be +40 N.m, and if it were bending it backwards, -40 N.m.

I've started to learn about PD (Proportional-derivitive) controllers. Robots like the H1 have a PD controller over a PID (-I-ntegral included) controller because their movements and actions are far too quick for an integral to matter. The integral is too slow, and it would likely lead to worsening the error for a fast-moving robot.

The PD controller sits between the policy and the motor drivers. The PD controller is fed by the policy, and it runs at a much faster pace to keep up with robot actions.

I've also learned that sim torque limits and real hardware limits don't always align. The built in articulation H1_CFG has the H1's hip, knee, and torso limits set to 300 N.m, whereas real hardware on the H1 has a limit of 360 N.m for the knee joints and 220 N.m for the hip and torso joints. The limits are set higher due to sim's 'perfect' nature, where it doesn't account for motor heat and speed. This does introduce a bit of a sim-to-real transfer gap, where the limits would ideally be tweaked to more closely aligned spec if transferring to real hardware.

There are also stiffness and damping values baked into the articulation config. How stiff a joint is determines how aggressively it snaps to a target. Higher stiffness means a joint snaps to a target quicker whereas low stiffness means a joint moreso drifts toward a target. High --> snap; Low --> drift. Damping values control the dampening (duh), which means higher damping values keeps the joint firmly on target and lower damping values allow oscillation from the target.

I have learned that the "Root" penalties listed in rewards.py of the mdp library literally means "root" as in full robot body. I may be an idiot, but at least I'm only an idiot some of the time.

## Run 012:

I've learned that the \_b suffix represents the "body frame", or the root of the humanoid. In the H1's case, the body frame is the link between the pelvis and torso. This body frame, used in root penalties, only shifts when the actual body frame moves. This is opposed to the CoM, as the CoM can shift with other joint movements (the H1 swinging its arms forward or backward).

## Run 013:

It is important to keep the relationship between the is_alive reward and the is_terminated penalty in mind when weighing rewards. Ideally, is_terminated should be > is_alive (full episode), and meaningfully so. If is_terminated isn't substantial enough, the policy will see occasional falls as just fine, whereas a fall should be 'catastrophic' and the most undesired behavior of all.

## Run 014:

PPO outputs a probability distribution. For continuous control of robot joints like this project, it is likely a Gaussian distribution. I remember encountering the Gaussian distribution when looking at IQ scores and how the general population lines up. We are likely working with a multivariate gaussian distribution, which takes multiple variables into account (since the H1 has lots of joints working in unison). This turns the typical 2D Gaussian distribution with its bell curve and stretches it to multiple dimensions, while retaining the same shape.
