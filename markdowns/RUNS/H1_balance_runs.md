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

## Run 004

### Hypothesis:

Although the ankle penalty likely needs tweaking, it will help to include the is_alive reward. The reward will change the policy itself, vaguely.

### Changes:

- Added is_alive reward with weight 0.05
- Increased iterations from 500 to a meaningful 1000

### Result:

Just adding the is_alive reward changed the learned behavior dramatically. By i300, the H1s were learning how to stay upright for longer, and by i700, they got really good at it. By the end of the run, the H1s sufficiently maintained 20s of balance ~70% of the time. Compared to the previous run (500 iters), this run managed to get to a 30% balance rate vs. 0% at i500. Although the balancing behavior is still erratic (shimmying and moving a lot), this is a pretty drastic improvement over run 003.

## Run 005

### Hypothesis:

The H1s have proven that they can stay upright and maintain "balance", but it is not in the usual, energy efficient sense. Some of the H1s in the past few runs travel quite far in their pursuit of stability. This works, and it can be chalked up as a form of reward hacking. A more ideal behavior would be a stiller version of what they are currently doing, remaining stable without much movement. One method of accomplishing this is incorporting the action penalties (action_l2 and action_rate_l2), which penalize large commands (action_l2) and rapidly-changing commands (action_rate_l2). This should prevent the H1s from at least travelling across the map to stay upright.

### Changes:

- Added action and action_rate penalties with weights -0.01

### Result:

One of the most fascinating runs so far. The H1s really stagnated in their balancing attempts early on, almost certainly due to the new action penalties. In earlier runs, the H1s would usually be moving and jittering around quite a bit by i500, but they hardly moved in this run. It wasn't until right at the very end of the 1000 iterations that they began to show meaningful signs of balance (of course). The main difference between this run and the previous ones is the scale of movements. Instead of spazzing out, the H1s in this run (~i999) gracefully shuffled their feet forward to maintain balance. This still led to the H1s greatly drifting from their original positions overtime, but it was an improvement over the erratic behavior that emerged from past runs.

## Run 006

### Hypothesis:

Adding action and action_rate penalties removed the spastic behavior learned in previous runs, allowing for a more ideal, slow balance. However, the H1s still drifted across the environment. The goal of this task is for the H1s to learn how to balance in place (stand up straight). By adding a flat_orientation penalty which penalizes the base for not being flat in relation to the ground plane, this will, over time, reduce how much the H1s drift in their pursuit of balance.

### Changes:

- Added flat_orientation penalty with weight -0.1

### Result:

Halfway through the 1000 iterations, the H1s learned an interesting behavior: bobbing. They stayed relatively still, minimally bobbing up and down to keep somewhat balanced. Unfortunately, this technique did not work, resulting in the H1 falling within 5-6 seconds. However, the H1s started learning how to stay balanced toward the end (i800-i999). Their bobbing technique shifted more into a "lean with it, rock with it" technique. The H1s would lean backwards, shifting their leg and ankle joints to not fall backwards. Occasionally, some H1s would start leaning forward, causing them to shift their leg and ankle joints forward instead. This was done in a more alternating step pattern (one step after another) as opposed to the default both steps back at relatively the same time. Usually, H1s that started balancing forwards would eventually lose balance and hit the ground (oof).

## Run 007

### Hypothesis:

The flat_orientation penalty did help mitigate some of the drifting behavior, but the H1s were still prone to it in nature. It would be really wise to add a custom penalty for H1s drifting away from their origin point, but I would like to experiment with some more built-in functionality first. By adding an angular velocity penalty, the H1s will be incentivized not to "lean with it, rock with it". The old technique may be reshaped to a more forward-backward-forward technique instead, or something new entirely. With a weight of -0.05, this penalty should weigh similarly to the others, leading the H1s to take all of them into account.

### Changes:

- Added angular velocity (tilt/pitch) penalty with weight -0.05

### Result:

Although the H1s failed to balance for a whole episode by the last iteration, the new angular velocity penalty on top of the current rewards actually eliminated the drifting problem. By the last iteration, the H1s were showing clear signs of attempted balance. However, they did not overcome their falls. Eventually, their torsos (and overall COM) would tip slightly forward or backward, leading to the H1 losing its balance and falling to the ground.

## Run 008

### Hypothesis:

Run 007 offered a close enough look at a potentially good run, so resuming it for another 1000 iterations is meaningful. The H1s will likely figure out how to stay upright given enough iterations.

### Changes:

- 1000 more iterations, resumed from run 007

### Result:

The H1s learned the stanky leg. Fascinating. The learned behavior differed pretty drastically from i999 to i1998. At i999, the H1s did not move their feet much, if at all. 1000 iterations later, the H1s shimmy their feet. Interestingly, the H1s developed a behavior where they effectively pivot in a circle. They first lean their torso to the left, over their left leg, then they keep their left leg sturdy with little movement while their right leg twists and shifts forward. This leads them into a perpetual counter-clockwise circle.

Most interestingly of all was one of the 32 H1s I watched closely in the i1998 .pt replay. Unlike the rest of them, this H1 (we'll name him Timmy) got unbalanced early on, leading him to almost tipping backward. However, Timmy still used the stanky leg method. This saved Timmy from falling, but his balance was still controversial. Timmy continued trying to pivot in the same counter-clockwise circle as his peers, but his movements and unstable balance prevented his right foot from moving forward enough to do so. It would usually freeze in place, leading to Timmy tilting backward. Instead of falling, Timmy would pause his counter-clockwise pursuit and shift his right leg back a step or two before trying again. It never prevailed, leaving Timmy in a perpetual backwards clockwise pivot instead. It is fascinating that this method, although clearly optimized for counter-clockwise rotation, still worked for Timmy going clockwise.

There is a philosophical metaphor here as well: No matter how hard you try to be like everybody else, the universe may have a different path carved out for you.

## Run 009

### Hypothesis:

Okay, that's enough philosophy. I have a better grasp on mdp's built-in rewards and how they function. While there may be a specific set of built-in rewards, with appropriate weights, that could better meet the goal of getting an H1 to balance in place, I believe the most efficient method of getting the H1s to actually balance in place would be a custom origin-drifting penalty. The torso would be a good starting point, as the COM generally sits in the lower torso when in a typical standing position.

### Changes:

- Added a custom torso_drift penalty with weight -0.1
- Temporarily removed all other penalties besides staying_alive and is_terminated

### Result:

Very erratic behavior has returned. Early on, the H1s learned to use vertical space (Z-plane, unpenalized) and a wide-legged stance to balance. However, by i999, the jumping technique became erratic enough to where some of them would jump a good ~0.5m off the ground and come crashing down. Leg movements and torso swiveling also became very severe and desperate. Overall, the H1s weren't able to learn a stable balance in the 1000 iterations.

## Run 010

### Hypothesis:

A couple things. I had assumed the H1s might use jumping as a technique, since I don't have any penalties against verticality. Adding a vertical penalty to supplement the torso_drift penalty will reduce jumping behaviors. In the previous run, the staying_alive reward also overpowered the single torso_drift penalty. By adding a similarly weighted vertical penalty, the two penalties will have a greater effect in the reward space. If not, the staying_alive reward may need to be reduced.

Also, as seen in the past few runs, the H1s tend to start learning toward the end of the 1000 iterations. Increasing to 1500 iterations should capture a better look at learning while also staying resource-efficient. It will allow me to see runs that may start showing signs of progress late (i999-i1200).

### Changes:

- Added vertical penalty with weight -0.1
- Increased iterations from 1000 --> 1500

### Result:

The H1s developed a unique strategy this time. By i300, they learned to straighten out their legs and start with a sideways torso, forearms flexed. This worked somewhat, but not long enough to balance the whole episode. Eventually, they would tilt too far one way or another and fall over. By i1499, the strategy proved even worse. The H1s got more wobbly and over-reactive, leading to even quicker fall overs. This may actually be a case of overfitting, despite the policy never fully learning how to balance.
