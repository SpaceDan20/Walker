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

I've started to learn about PD (Proportional-derivitive) controllers. Robots like the H1 have a PD controller over a PID (-I-ntegral included) controller because their movements and actions are far too quick for an integral to matter. The integral accumulates over time, so if a robot joint overshoots or gets perturbed, the integral would overreact and cause instability. The policy itself takes over for the integral, effectively learning how to correct errors throughout an episode.

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

## Run 016:

An L2 norm is preferable in the case where a threshold and clip is given, since it can _quadratically_ penalize an undesired behavior the further it goes past the threshold.

## Run 018:

Each run has a params folder set by the policy (RSL-RL in our case) with a few yamls inside. The tensorboard metrics from run 16 & 18 perfectly match because the same seed was used in both yamls (seed=42). If I wanted variation in a run, I would change the seed before starting it. Using a seed means the training is deterministic, which is the preferred way of training due to its advantages in reproducability and ablation (testing different reward configs with the same seed to verify that the reward config is the determining variably). Deterministic training also means that I can test a policy for robustness by training a few runs with different seeds (run 19 could be seed 43, run 20 could be seed 44, etc.).

## Run 019:

Reward shaping is, by far, the most important thing when it comes to RL learning. Overtime, I have learned just how much the reward config can make or break a policy. Up until now, I haven't given reward structuring enough thought. I learned early on that there is no specific formula that plugs into the whole reward landscape, and that even the most renowned RL researchers experiment with hand-crafted reward configs. There are, however, good practices to take into account when designing a reward landscape. It is best to work backward when designing a reward, so I would:

- First, find out what a perfect episode should look like magnitudally (is that a word? it is now). For instance, I could say that a perfect episode for balance should accure a roughly +20 reward.
- Next, the actual 'reward's. This would be just the staying_alive reward. I would determine how much it should influence the landscape. Ideally, the staying_alive reward would be doing the brunt of the work, so we'll say +15 maximally.
- ~~Then, we have to determine the magnitude of staying_alive's evil twin, is_terminated. is_terminated should be more than the staying_alive reward x max_steps so that falling over feels consequential instead of slightly annoying. In this case, is_terminated could be -20 since (20 > 15). I am so good at math.~~
- Finally, penalty weights must be determined. The penalties should be prevalant but not overpowering the staying_alive and is_terminated rewards. Their meaning is to give the policy a gradient signal to follow, that's it. So the penalties must be weighted as a sum, appropriately. In this example, we could determine that all of the penalties should accumulate no more than -10.

Using these magnitudes would be significantly better than blindly guessing what numbers would work in the landscape. Following this general 'formula' is the best way to making a healthy reward config that works.

## Run 021:

Watching replay (.pt) files and analyzing TensorBoard metrics are essentially just the floor of true RL research. These two, especially with my custom focus_play script, have been informative of my design choices, but they are simplistic in nature. The .pt files are full checkpoint files loaded with all of the information necessary to capture much more detailed data.

A very useful diagram would be the phase portrait. A phase portrait is a 2D visualization of a trajectory(s). By plotting the sagittal plane (the humanoid's pitch) and the frontal plane (the humanoid's roll), I can further diagnose what trajectories the H1s are taking without having to visually inspect them. This allows for more precise diagnosis, as I can clearly visualize the phase portraits without the need of tracking a moving body. It will also help for later runs, where the H1 will inevitably show less and less notable movement as a good future policy stabilizes.

Sensor noise (observation corruption in sim) is very consequential. The same checkpoint (run 21's model_500.pt) run with and without observation corruption yield very different results, despite both playbacks being deterministic (inference model). Running the .pt file with focus_play (currently no observation corruption) shows the H1 converging to an incredibly still, unmoving pose, with torques freezing to within the hundredths (knee joint torque freezing at ~68.73-68.74, for example). However, when playing the .pt file using play (with observation corruption), the H1s are much less stable and exhibit far greater motion overall.

Policies are always run deterministically (at deployment). Stochastic sampling only happens during training, where the policy (PPO) uses a Gaussian distribution to sample actions. If it didn't, the policy would not be able to learn by exploration (picking random actions instead of the best guess early on). Once a policy is trained, deterministic inference is used because the best guess is desired over random actions.

## Run 022:

Shaped penalties come in all shapes and sizes, and they hold meaningfully different purposes. These should be taken into consideration when determining the weights. Ideally, the shaped penalties are led by more purpose-driven penalties, such as flat_orientation for a stability task. Meanwhile, more efficiency-based penalties like action and torque penalties should be weighed less. This creates a soft curriculum effect, where the policy will more likely minimize the purpose-driven penalties first. Once these penalties are taken into account, hopefully teaching the agent the intended behavior, then it can start focusing on the less important, yet still helpful penalties.

## Run 025:

When it comes to humanoids, and humanoid locomotion training, curriculums are the norm, not an exception. The right framing around curriculums isn't whether or not to include one - it is which form of curriculum would be best suited for the task.

## Run 026:

The value function loss is heavily influenced by the magnitudes of rewards and penalties. The critic (value network) has the singular job of predicting, from a current state, what the total amount of accumulated reward for the rest of the episode will be. The value function loss is the MSE (mean squared error) of (the actual calculated value of the decision step - the critic's prediction). Early on, when the critic hasn't learned anything, it typically outputs values close to 0. This means that the MSE for massive rewards/penalties is enormous. For my experiment of setting the termination penalty to -100,000,000, this generated value function losses in the quadrillions since the outcome was (~-100,000,000 - 0)² ≈ 10,000,000,000,000,000.

PPO uses an Actor-Critic framework. The actor and critic are seperate networks with different objectives. The actor acts on the environment and the critic predicts the value function (discounted rewards accumulated for the rest of the episode). The critic plays a role in updating the advantage function so the actor can learn from episode to episode.

## Run 027:

I have learned that my original understanding of good practice reward magnitude structures is incorrect. I was going off of the premise that my discrete termination penalty should be higher than my maximum survival reward for the episode, whereas the correct logic was having a termination penalty that is higher than a single step of survival. Generally, discrete termination penalties like mine take up a small to tiny portion of the reward landscape (think 10% instead of 50+). With my high termination penalty in the previous runs, it ended up overshadowing the crucial staying_alive reward once they both "cancel each other out". Tensorboard showed me that this occurred once the H1s reached about a ~60% survival rate. At that point, the termination penalty was responsible for ~-0.04 reward per step while my staying_alive reward was giving ~0.04. It isn't that simple, however. This is mainly an issue with the critic struggling to find "slightly better states" when the return estimates converge close to 0 (at around this 0.04 point). The critic is essentially blinded by the extreme termination penalty, unable to see the tiny gradient signal provided by the shaped staying_alive reward. This is the reason the H1s continued to slowly improve their survival rates (60% --> 75% in 800 iterations) instead of completely flatlining. Therefore, I suppose, the massive termination penalty method does work, albiet extremely inefficiently.
