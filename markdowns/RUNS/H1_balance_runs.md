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

Over-reactive may not be the best term to describe the learned behavior. By adding a torque tracker, I can confirm that many of the joints are being locked up/maxed out by the policy. Hips and knees especially are continuously sitting/saturating at their 300 N.m sim effort limits, as defined by the H1's asset dicts inside of unitree.py.

## Run 011

### Hypothesis:

The current rewards are a good baseline, and I do believe each of them plays a good part in minimizing what behaviors the robot should not perform. When it comes to balancing, generally the upper body isn't really thought about (minus the human head for the vestibular system). However, robots don't have an inner ear. Most of their balance would come from the lower extremeties (mainly ankle and knee joints, with supplementary hips). I believe another good baseline penalty to add would be to penalize upper body movement heavily. This can be done using l1 vs. l2, as l1 penalizes small movements more aggressively.

### Changes:

- Added l1 upper-body joint velocity penalty with weight -0.05

### Result:

The H1s really want to do a backflip. For some reason, the policy learned to immediately kick hard and flip itself upside-down. This led to extremely short episodes (24 average steps, ~0.5s of time). This is, obviously, not ideal. On a positive note, the upper-body penalties seemed to work; The upper-body joints did not move at all. The problem seems to be an over-reactive and extreme leg response.

Another interesting thing to note would be the penalties. The policy slowly minimized every penalty except for the torso_drift penalty, which got very, very, very slightly worse over time.

## Run 012

### Hypothesis:

As cool as backflips are, they are not the goal here. Stable, human-like balance is. Given the fact that the upper-body joints were relatively fixed in the last run, the upper-body penalty seems to be working and should stay. The main issue is extreme leg movement; enough to literally send the H1 into a backflip. To stop this behavior, we can add back the action penalties. The action_l2 penalty, in particular, should prevent the H1 from wanting to commit large N.m of force suddenly and explosively.

### Changes:

- Readded the l2 action penalties.

### Result:

The H1s want to do the flop now. Instead of the aggressive backflips from run 11, where the H1s usually landed upside-down on their heads, the H1s flopped on their backs in this run. It followed the same general behavior, though: move right foot forward slightly, press it into the ground, and finally use the hip joint (pitch in particular) to drive the leg into the ground, causing the H1 to flip backward. The flops were learned relatively quickly, within the first 100 iterations. Closer to the end (i999), the flops got a bit slower, allowing the H1 to survive a very brief moment longer than in early flops. Similarly to the past few runs, the torso_drift penalty was the only penalty that was, for the most part, ignored.

## Run 013

### Hypothesis:

The action penalties seemed to help, albeit loosely. The strange "kick yourself backward" behavior did start off less aggressive, and eventually slowed down. Judging from the tensorboard graphs, it seems like a reward weight restructing is necessary. The torso_drift penalty probably should not sit at half the weight of the other, simple penalties. Upping the torso_drift penalty to -0.1 is a good start.

Also, the termination penalty is largely useless in its current state. When it was the only reward signal in the space (run 001), it actually helped the H1 learn how to roughly balance. Adding rewards on top of this penalty isn't the inherit problem - it is adding rewards without adjusting the weights properly. Now, with 4 other continuous rewards fighting for gradient attention, the -1.0 discrete termination penalty is effectively white noise, if that. Largely increasing the termination penalty should drive more desire for the H1s to not flop onto the floor.

These two penalty updates should balance the reward landscape and give the H1s a much cleaner signal to learn ideal behavior from.

### Changes:

- Increased torso_drift penalty from -0.05 --> -0.1 to match other continuous penalties
- Increased termination penatly from -1.0 --> -10.0

### Result:

The flopping behavior continues. The behavior is still learned early, still follows the same exact behavior learned in the past two runs, and still "slows down" over time.

## Run 014

### Hypothesis:

The H1s discovered the backflip in run 11, when the l1 upper_body penalty was added. Ever since then, they can't get enough. They want to backflip/flop very badly. They are so agitated that they are punished for moving their upper body that they simply can't go on. However... they must. Although removing the upper_body penalty would likely result in completely new behaviors being learned, first readding the flat_orientation penalty might mitigate the flopping behavior by penalizing the tilt that occurs in them.

### Changes:

- Readded flat_orientation penalty with weight -0.1

### Result:

No luck. The backflip must be done. This will be a fantastic starting point for my eventual backflipping task. They're already halfway there! All of the penalties followed a similar trend from the past few runs, and the flat_orientation penalty joined the torso_drift penalty in being largely ignored by the policy.

## Run 015

### Hypothesis:

Although there are many variables to take into account, and the recently readded flat_orientation penalty was mostly ignored after the first ~100 iterations, the l1 upper_body penalty is likely to blame for the backflips. A critical issue is that the termination penalty sits at -0.01 (per step), whereas early on, when the flips are learned, the upper_body penalty accumulated as much as -0.07. The penalty is far too strong, and it quickly throws all the other penalties into the void while the policy aggressively minimizes just it. This is also why the torso and upper body all quickly learn to be relatively still as early as i100 (despite tilting due to the backflip/flop).

### Changes:

- Drastically reduce the l1 upper_body vel penalty from -0.1 --> -0.01 (x10)

### Result:

The backflips/flops have been overcome. Although the H1s started demonstrating the same backflip/flop behavior in the first 100 iterations, it was quickly dissapated by i200. The penalty graphs between this run and last look meaningfully different, where most trends breakaway around i100. Interestingly, all penalties started to increase toward the end, with mean_reward dropping, despite the average episode length steadily increasing. By i499, mean_episode_length was increasing substantially, and the H1s were getting close to balancing the whole episode.

The new behavior is getting very close to the goal. The H1s aren't drifting from their origin point like in the first ~10 runs, but they are actively showing signs of balancing. At i499 (last iteration), they do appear to be bending at the knee a bit more than intended. It is worthwhile continuing this run to see if they develop a crouch-like stance to balance, which is undesired behavior but still a step-up from the early, origin-drifting runs.

## Run 018 (016 & 017)

### Hypothesis:

The H1s are very close to the stable, human-like balancing behavior we are looking for. The only suboptimal behavior they are demonstrating is a crouch-like stance, bending too much at the knees. This behavior can be penalized by using a custom l2 knee_bend penalty, which calculates a specific threshold (30 degrees in this run's case) and penalizes the knee for exceeding it at an exponential rate. This penalty should teach the H1s to keep their legs relatively straight, finalizing the balancing behavior.

### Changes:

- Added knee_bend_penalty with weight -0.1

### Result:

The H1s learned to keep their legs relatively straight, but at the cost of falling over backwards. The new knee_bend penalty was minimized to nearly 0 up until about i420. At about that iteration, the policy really started to ignore all of the penalties. This is because they learned the same crouch-like stance from last run by ~i500, which extended their average episode length by 2x. Mean reward did drop, however.

Around ~i700, the H1s started to minimize penalties again, attempting to return to the same magnitudes as earlier iterations (200-300). In particular, they stopped crouching and started keeping their legs straight once more. This, unfortunately, led to worsening balance, and the episode length declined to even worse levels than before (~100 --> ~80-90).

## Run 019

### Hypothesis:

The H1s did take the new knee_bend penalty into account, but vaguely. When it comes to the other penalties, the knee_bend penalty is hardly weighted. At its worst, the penalty (per step) racks up in the tens of thousandths (-0.0001), whereas the other penalties live in the thousandths range (-0.001). By upping the knee_bend penalty by x10, the policy will weigh it the same as the other penalties.

Also, the torso_drift penalty appears to be weighted too much. It is the only penalty that dropped out to -0.01 per step when the policy temporarily quit minimizing its penalties. This was also the time where episode length was at its highest point. By reducing the torso_drift penalty, the policy can focus more on minimizing other penalties and allow for more liberal movement of the torso to balance.

### Changes:

- x10 knee_bend penalty from -0.1 --> -1.0
- -x2 torso_drift penalty from -0.1 --> -0.05

### Result:

The H1s followed roughly the same learning curves as the previous runs (runs 15-18). Same policy collapse around halfway through the training (~i400-500), and roughly the same metrics overall.

## Run 020

### Hypothesis:

The reward landscape needs more work. The first major misstep I can spot would be the staying_alive versus is_terminated rewards. In the past few runs, the policy has collapsed when the staying_alive reward starts to grow substantially. For the training and physics setup, a maximum episode length is 1000 decision steps. At the current weight of 0.05, the staying_alive reward can accumulate a +50.0 signal. This greatly exceeds the one-time is_terminated penalty of -10.0. I had learned about the importance of these two magnitudes around runs 12-13, but I neglected to do the math necessary to make sure that (is_terminated > staying_alive x max_episode_length). Fixing that alone may cause the policy to not collapse halfway through training.

### Changes:

- -x10 staying_alive reward from 0.05 --> 0.005

### Result:

The H1s learned to vaguely balance, but not optimally. The policy took a bit longer to "collapse", leading the H1s to steady, in-place balancing until ~i600. At i600, the H1s began ignoring penalties to stay upright for longer. They developed a walking behavior once more, where they would drift far away from their origin to remain upright. They mostly ignored the torso_drift penalty, by a large margin. Despite mean_reward declining in this period, episode_length similarly rose. Eventually, the policy took penalties back into account.

Most interestingly in this run, by i999, the H1s captured a mixed behavior of the early and middle iterations. The H1s learned to mainly stay in place, and when they started losing balance, they would shuffle their feet slightly opposed to outright walking forward. This reduced the torso_drift penalty, as it kept them moderately balanced while keeping their origin closer.

Despite this, the H1s did not achieve meaningful balance in the 1000 iterations.

## Run 021

### Hypothesis:

Until now, I have not put enough thought into the reward magnitudes of this config. The earlier runs were moreso a learning experience of mdp's built-in rewards, as well as how rewards are built inside of IsaacLab. Now that I have a better grasp of that, I need to logically restructure my reward magnitude landscape, as the current magnitudes are suboptimal.

Restructuring the reward landscape by working backwards is the best bet to enhancing this task. I must first determine what a "perfect" episode, as well as "decent", "suboptimal", and "horrible" episodes look like. The magnitudes matter in this regard. Currently, every reward in the config run at roughly the same scales (~0.001 for the shaped rewards).

Firstly, the H1s don't have nearly enough of a signal from either staying_alive or is_terminated. is_terminated is greater than staying_alive x max_steps, but staying_alive is not nearly meaningful enough. At 0.005 reward per step, the staying_alive reward only rewards a max of 5 for a fully survived episode. I will start by upping this max to 20 (0.02 per step). The is_terminated penalty needs to be greater than this, so I will increase it to be higher as well (-30 for a fall). -30 is a large enough penalty to substantially discourage falls.

The shaped penalties were also too strong in the previous config, but the staying_alive and is_terminated magnitude changes do bring them closer to the gentle gradient signal they need to be. I will still decrease the penalties to more appropriate, gentler signals in the thousandths range. With 7 shaped penalties accumulating reward together, even penalties in the -0.002 per step range may be too high. -0.0015 to -0.002 should be a good starting point, though.

### Changes:

- Greatly increased the staying_alive reward from 0.005 (5.0 max) --> 0.02 (20.0 max)
- Greatly increased the is_terminated penalty from -10.0 --> -30.0
- Decreased shaped penalties from -0.1 and -0.01 to the -0.0015 to -0.002 range (1.5 to 2.0 maximums)

### Result:

I may be closer to my dancing project than I once thought. This run was drastically different than any of the other runs, by a long shot. Somehow, throughout the 1000 iterations, the H1s developed as many as 4 or 5 unique strategies to balance. The run was looking pretty healthy up until halfway through, where the policy began to collapse. After it collapsed, many interesting behaviors replaced the previously good balancing strategy. As a summary:

i1 - i500: The H1s learned to balance reliably, bringing episode termination by falls all the way down to 5% (95% stochastic survival rate). Unfortunately, they did so by utilizing a wide, crouch-like stance (not the goal). Either way, it only took them 200 iterations to reliably balance this way.

i500-700: Policy begins to collapse. The H1s continued their crouching approach, but they began leaning their bodies too much. Penalties started to accumulate substantially, dropping mean_reward from ~-1 to -4 by i700.

i700: Policy collapse. The H1s still crouched, but they developed a strategy of leaning their torsos completely back (horizontal to the ground) and using their bent legs to hop backwards. This led to incredible origin-drifting behavior, where the H1s ended up very, very far away from where they started (obviously not ideal, this is not a travelling task!)

i800: A dark force has awakening inside of the H1s. They finally got away from the leaning backward strategy, instead developing a new strategy where they raise their left arms in a controversial manner. The crouching improved, but they retained a wide stance.

i999: I'm not sure what you would call this. The wide-legged stance remained, crouching still wasn't bad, but the upper body had an interesting approach. The H1s appeared to return from the dark side, instead raising a bent left arm. As for their right arms, they had them slightly tucked down behind their torso, with their elbows bent and their forearms raised upward. Their torsos were ever-so-slightly bent and twisted.

This is probably the most interesting policy collapse I've witnessed so far. It will take some more in-depth research to understand just what happened, and how the gradient signals took hold of this run.

## Run 022

### Hypothesis:

The reward magnitude changes certainly took effect. staying_alive and is_terminated are regarded very strongly in the policy, collapse or not. While these may need to be tuned down, it makes sense to first fix the shaped penalty magnitudes. Setting all of the shaped penalties from -0.0015 to -0.002 doesn't work when some of the penalties trigger almost all of the time while some trigger infrequently. To start out, it would be reasonable to reduce the action penalties since they dominated the shaped penalty signal, and the H1s need to be unafraid to move for the sake of balance. Also, arguably one of the most important shaped penalties available (flat_orientation) is currently underutilized, not even clocking in -0.001 per step. Increasing this penalty should help the overall gradient signal.

### Changes:

- Greatly reduced (-x10) action and action rate penalties from -0.0015 to -0.00015
- Greatly increased (x10) orientation penalty from -0.002 to -0.02

### Result:

The policy did not collapse this time. It moreso morphed its undesired behaviors into different undesired behaviors. The undesired behaviors followed a vague trend: wide-legged stance --> slight crouch --> raised arms --> twisted torso --> twisted arms. Throughout the run, the H1s struggled to stay near their origin points as well. The major improvement would probably be the smoother metrics and the fact that the H1s simply plateued at an ~80% survival rate instead of rising to 95% before intensely dropping out twice later in training.

## Run 023

### Hypothesis:

First, we must address the origin drifting behavior. Despite falling over, this may be the single biggest undesired behavior. Using a wide-legged stance to balance? Understandable. Raising arms to do so? Sure. Leaping backward across the map? Where are you going?! There are currently two shaped penalties contributing the most to stop the origin drifting behavior: the torso_drift penalty and the upper_body_vel_penalty (vaguely). The current torso_drift penalty is meaningful, but not when it is stacked up against the action penalties. Despite being nerfed in the last run, the action penalties are still contributing largely to the gradient signal. The action penalty alone is accumulating up to -0.04 reward per step toward later in the training. I've learned that the action penalties, which can be labelled as 'efficiency' penalties, should be weighed much less than goal-oriented penalties like torso_drift or flat_orientation. By once again nerfing the action penalties, we should see the policy take these goal-oriented penalties into account more effectively.

Also, the relationship between arguably some of the most important shaped penalties should be addressed. Currently, the upper body velocity penalty dominates early training iterations. torso_drift does catch up, but it takes some time, and flat_orientation doesn't hold a candle to either. Ideally, the main driver should be torso_drift (to stop origin drift), secondary should be flat_orientation (to promote balance), and the last should be upper body velocity penalty, as it is moreso an efficiency penalty. This weighing should capture more meaningful behaviors while discouraging undesired ones.

### Changes:

- Greatly reduced action and action rate penalties from -0.00015 to -0.000015
- Slightly increased torso_drift penalty from -0.002 to -0.0025
- Greatly increased flat_orientation penalty from -0.02 to -0.1
- Greatly reduced upper_body_vel penalty from -0.002 to -0.0005

### Result:

As with the previous runs, some undesired behaviors inevitably showed up. Mainly wide-legged stance and slight crouching, but we aren't penalizing for that just yet. The H1s only managed to achieve a ~65% survival rate in the 1000 iterations. Despite it being more stable than previous runs (staying firmly between the 60-80% bars), this is not an ideal survival rate. On a more positive note, the shaped penalties rebalancing did take "shape". Sorry. The action penalties are no longer dominating the signal, allowing torso_drift to shine.

Vaguely. I have new origin drifting graphs to go off of, and the trend isn't great. i200 is a good starting point, where H1s started getting good at surviving. Between i200 and i300, the H1s drifting behavior improved drastically -- down from 5m average horizontal drift to 0.6-0.8m. i400, it increased to ~1.2. i500 ... back to 5m. i600 to i900, the H1s started to explore the world, with drifts increasing to 10m+. The last iteration, i999, showed improvement, back down to 4, but the policy still clearly had an interest for exploring the world.

## Run 024

### Hypothesis:

I think the biggest issue of the last run was the termination rate. The H1s started to get a relatively strong gradient signal from the shaped penalties before they could achieve 80%+ survival rates. The policy started minimizing these penalties at a 65%-70% survival rate. This is suboptimal. The is_terminated penalty needs an increase so the H1s first learn how to survive, ideally, at least 90-95% of the time. Only then should the shaped gradients be strong enough to encourage minimization. As for the origin drifting behavior, the shaped penalties may need more tweaking. We must first get the survival rate back up to 95%+, even if it means the H1s explore the world temporarily. In the meantime, reducing the still-too-powerful upper body velocity penalty will strengthen the overall shaped penalties gradient.

### Changes:

- Greatly increased is_terminated penalty from -30.0 to -75.0
- Moderately increased staying_alive reward from 0.02 to 0.025
- Greatly reduced upper body velocity penalty from -0.0005 to -0.0002

### Result:

The H1s are definitely still coming up with interesting strategies to keep balanced. The behavior from the last iteration (i999) can best be described as someone who has to pee while they simultaneously cough into their arm. It is a very strange pose. Either way, the H1s stagnated at about 75%, which is when the gradient signal started getting messy. They never fully learned to survive, and they never learned a stable, still balancing behavior. They still have wide-legged stances, they still crouch, they still drift away from their origin points, and they still do weird stuff with their arms.

## Run 025

### Hypothesis:

Simple reward magnitude tweaking won't fix this behavior. i400 of the last run was the most stable checkpoint, and even then the H1s were drifting and demonstrating undesired behaviors. Instead, a more generalized return of a shaped penalty I took out, joint_deviation_l1, should directly target this problem. With its addition, the upper body velocity and knee bend penalties aren't all that necessary anymore. Removing them will allow more room for the other shaped penalties to breathe, especially this new joint deviation one. The key is to implement it in a healthier shaped penalty field. With its addition, and some shaped penalty tweaks, the H1s should stop: crouching with a wide-legged stance, origin drifting, and making unnecessary upper body movements.

### Changes:

- Added joint_deviation_l1 penalty with weight -0.001
- Removed upper_body_vel penalty
- Removed knee_bend penalty
- Increased is_terminated penalty from -75.0 to -100.0
- Increased staying_alive reward from 0.025 to 0.05 (50.0 max)
- Decreased torso_drift penalty from -0.0025 to -0.002
- Increased orientation penalty from -0.1 to -0.12

### Result:

The H1s are rebellious. They kept on crouching, they kept on with their wide-legged stance, they kept on drifting from their origin, and they kept on making unnecessary upper body movements. However, there still were improvements. They drifted less, they tilted less, and they rolled about the same yet more stable. They did not manage to get reliably above 70% survival rates, though.

## Run 026

### Hypothesis:

There are many issues to overcome. The main issue would be the suboptimal survival rates later in training. The H1s learn to survive pretty quickly, but their survival rates start to plateu at a suboptimal 60-70%. This is due to the shaped penalty gradient signals appearing much larger once the massive termination rate weakens from survival. Essentially, the H1s are trying to optimize their shaped penalties before adequately getting a reliable 95%+ survival rate.

Most approaches to solving this problem incorporate some kind of curriculum, allowing the robot to learn concepts one at a time by usually balancing weights. This would be done by starting off with extremely low shaped penalties, then ramping them up later in training once the H1s can survive.

However, I'd like to try something different first. Before implementing a curriculum, I am going to try rebalancing the penalties to the extremes. I will let the H1s train with an enormous termination penalty and microscopic shaped penalties, just to see what happens. The staying_alive reward will also accompany these microscopic penalties, to a slightly higher degree for influence. This approach should develop unique behavior, whether or not it is ideal.

### Changes:

- Increased termination penalty from -100 to -100,000,000
- Decreased staying_alive reward from 0.05 to 0.025

### Result:

Flatline. The outrageous termination penalty completely decimated any attempts at learning. Setting the termination penalty to -100,000,000 led to the early critic, with its initial weights close to 0, struggling to fit the scalar value. The value function loss exploded in the quadrillions, only coming down to the billions after ~300 iterations. This is where is flattened out, unable to fit the enormous termination penalty adequately.

## Run 027

### Hypothesis:

While it has been fun tweaking rewards and their respective magnitudes, the only real way to solve this plateuing survival rate problem is a curriculum. By incorporating a curriculum that scales down shaped penalties until a certain point (in our case: survival rate), the H1s can focus on surviving meaningfully before minimizing the shaped penalties. This should lead to a more natural advancement: learn not to fall --> learn to survive --> learn to balance.

### Changes:

- Added a custom reward weight curriculum that scales shaped penalties down to 5% until 95% survival rate can be reliably achieved.
- Brought termination penalty back to Earth (-100)
- Increased staying_alive reward back to 0.05

### Result:

In the 1000 iterations, the H1s did not advance the curriculum. The survival rate climbed predictably until about ~60%, where it started to plateu. This happened around iteration 350, where the termination penalty (-100) started meeting with the staying_alive reward (+50) at ~0.04 reward/step. Between i350-1000, the policy did steadily climb from this 60% to ~75%. Overall, the H1s learned to balance, partially, but at the cost of the optimal behavior (still, non-moving balance).

## Run 028

### Hypothesis:

Up until now, I have worked with the flawed reward magnitude logic of (is_termination penalty > maximum staying_alive reward). This was incorrectly interpreted by me. The logic I was provided was a basic reward magnitude guideline of (is_termination penalty > a single step of staying_alive reward). In practice, most policies implement a small is_termination penalty, whereas mine dominates the landscape. This causes the gradient signal to collapse once estimated returns approach zero (staying_alive reward catches up to the is_termation penalty). In the previous run, this was around a 60% survival rate. The survival rate then plateus as the critic network struggles to find the tiny gradient signal while the termination penalty keeps screaming in its ear.

Although this doesn't fully stop the critic from learning, it slows it to a snail's pace. It took ~100 iterations for the survival rate to climb from 0 to 60%, but then another ~600 iterations for that 60 to reach 75%. So, it works, but it is incredibly inefficient. This would be like watching a baby start learning how to stand up within a few years, but then continuously fall over for 8 more. At some point, the parents might want to try a different strategy.

Therefore, the H1s survival plateus should be fixed by simply reducing the termination penalty to a reasonable percentage of the overall reward landscape. The staying_alive reward must dominate, NOT the termination penalty.

### Changes:

- Dramatically reduced termination penalty from -100.0 --> -2.5

### Result:

Unfortunately, the ~60% survival rate plateu persists. The trajectories were similar to the previous run, absent the termination penalty. The H1s developed an interesting strategy that involved tucked arms, crouched bodies, and overall drift. They did not learn still, unmoving balance, but more importantly, they did not surpass an 80% survival rate in the 1k iterations.

## Run 029

### Hypothesis:

There is one interesting tensorboard metric that needs to be addressed: Entropy Loss. For the past 7 runs (excluding the monstrous termination penalty experiment), the entropy loss has steadily risen throughout them. It initializes at 27 (roughly normal range for an H1 with 19 continuous joints) and rises to the 40s for each and every run. The last run was the worst offender, climbing to almost 50. The last time the entropy was relatively healthy (steady decline to small positives) was run 019, with run 016 being the last truly healthy entropy (did not fall into the negatives, suggesting overcommittal). These runs were chaotic and suboptimal (bad reward magnitudes, low to no survival rate achieved), but they are worth noting.

I have yet to touch the entropy coefficient in my PPO config. It has sat at 0.01 for all 28 previous runs. Now, with a presumably healthy set of rewards with appropriate magnitudes, it is time to look into other discrepancies. It is very likely that this 0.01 coeff is far too high, encouraging much greater exploration than necessary. Reducing it should warrant a declining entropy as opposed to an unhealthy, forever-climbing one.

### Changes:

- Dramatically reduced entropy coeff from 0.01 --> 0.001

### Result:

Success! For the survival rate, at least. This is the first run that has acheived a stable, sustained 95%+ stochastic survival rate. The only other run that held a candle to this one was run 021, and that was just a brief 96% survival rate window before ultimate policy collapse. No other run has reached these heights. Most of the previous (more recent) runs have plateued between the 60-80% rates. Because of this success, the policy was able to advance the curriculum and begin minimizing penalties.

Unfortunately, the intended behavior of still, unmoving balance was not yet achieved. The H1s still exhibited undesired behaviors such as crouching, using a wide-legged stance, bending arms in unusual ways, and jittering around too much. Despite these downfalls, this run is a huge leap forward toward stable balance. Now, we can focus our attention more on the shaped penalty landscape.

## Run 030

### Hypothesis:

Lowering the entropy coeff helped clear the way for real learning to take place. However, there is still room for improvement. Lowering the coeff a little more should yield a healthier curve that approaches low positives quicker without overshooting.

Along with a slightly lower entropy coeff, some reward tweaking can now be done to minimize the undesired behaviors. I believe the hierarchy of shaped rewards importance is as follows: staying_alive ---> joint deviation --> orientation --> torso_drift --> verticality --> action penalties. Currently, orientation is too low, torso_drift is too high, verticality is too invisible, and the action penalties as well as the joint deviation penalty could use a slight boost.

Lastly, the curriculum advanced at a decent pace. However, upping the requirement for advancement from 95 to 97% should allow for a more stable transition.

### Changes:

- Reduced entropy coeff from 0.001 --> 0.0007
- Tweaked shaped penalties
- Increased curriculum advancement from 95 --> 97%
- Increased iterations from 1000 --> 2000

### Result:

The H1s are really, really good at survival now. We are achieving 99% sustained survival rates towards the end of the 2k iterations. However, the undesired behaviors remain. In this run specifically, the H1s really abused a wide-legged stance and crouching. Their movements were not chaotic; their joints handled small corrections much better than previous runs. The main challenge now is the crouched, wide-legged stance.

## Run 031

### Hypothesis:

We are getting closer to the goal. A lot of the kinks have been buffed out. This puppy should run real good, very soon. There is still some leeway to lower the entropy coeff, and still some reward tweaking to do. So far, I have handcrafted my rewards by eyeballing tensorboard metrics and viewing runs and charts. This is good, but I need a closer look at which terms are really contributing to the overall gradient signal. For this, I have added a custom tensorboard metric which tracks mean and std per term. When a reward term reaches high mean and low std, it is not contributing much of a gradient. The opposite is true for low mean and high std. With this new metric, I can more clearly see which terms are being "seen" by the policy at what points, which will lead to better decision making when it comes to magnitude adjustments.

### Changes:

- Added new tensorboard metric to view mean and std for reward terms
- Reduced entropy coeff from 0.0007 --> 0.0005
- Slightly reduced torso_drift penalty

### Result:

On a good note, the entropy coeff is looking very healthy now, and the new tensorboard metric is pretty useful. I am able to see that the staying_alive reward continued to contribute meaningfully to the overall gradient (50%+ throughout the run, with a few dips), which means that survival was never truly optimized. However, it did hover at 95%+ survival rates for an extended period of time. Unfortunately, survival started weening off at the end as the H1s began losing balance and falling backward. To top things off, it was not ideal balance anyway. Same wide stance. Same crouching.

## Run 032

### Hypothesis:

There is still work to be done. First and most importantly, there is a 'bug' in my torso_drift penalty. The H1s currently have spawn randomization (their base spawns up to 0.5m away from their origin in the xy plane), and the penalty compares against the env_origin, which is wrong. H1s which spawn away from this env_origin can then be influenced to step back toward it to minimize the currently broken penalty. Fixing this will allow the torso_drift penalty to contribute to the gradient meaningfully instead of harming it. It is wise to make just this change to see how it affects the training.

### Changes:

- Fixed the custom torso_drift penalty logic
- Added W&B logger for better data visualization

### Result:

Disaster. Everything was going smoothly, for survival, up until ~50 iterations after curriculum advancement. Then the H1s dropped to an astounding 0% survival rate for almost 100 iterations. The H1s also never developed a sound, ideal balancing strategy.

## Run 033

### Hypothesis:

The curriculum is almost surely to blame for the collapse. Currently, my curriculum allows the H1s to maximize their staying_alive reward and minimize the termination penalty until they can adequately survive. Then, after curriculum advance, the agent is hit with a whole new gradient landscape altered by an immense arrival of penalties which suddenly reached their full weights.

This isn't ideal. A smoothed curriculum may be a better fit, but it would be wise to first remove the curriculum without changing the weights to see the differences between the runs.

### Changes:

- Removed the curriculum

### Result:

Still not a very healthy run. The H1s occasionally learn how to survive to extreme lengths (95%+ timeouts), but they eventually drop dramatically. More importantly, the H1s still demonstrate suboptimal behaviors throughout the iterations.

## Run 034

### Hypothesis:

Not enough attention has been given to the observation space. Most notably, the agent cannot minimize my custom torso_drift penalty due to it not being able to perceive it through the current observation terms. The torso_drift penalty is also probably unnecessary over more helpful and relevant penalties like the joint_deviation penalty. Removing the torso_drift penalty in favor of the joint_deviation penalty should encourage less suboptimal behavior. In addition to this, adding the base_lin_vel observation makes the vertical_penalty work. A new base_height observation will give the agent helpful height information, and finally, a foot_contact observation will let the agent know whether or not both robot feet are planted, which should help it learn stable positioning.

### Changes:

- Removed torso_drift penalty
- Increased joint_deviation and action penalties
- Added base linear velocity observation
- Added base height observation
- Added foot contact observation

### Result:

Total success. The task has been accomplished.
The H1s finally learned how to survive ~100% of the time without suboptimal behaviors. They jitter slightly and drift subtly over time, but they reliably balance for the entire episode.

## Run 035 (Seed 2)

### Result:

Spoke too soon! Another run with a different seed prompted a less successful outcome. Survival spiked to 100%, dipped around i800, and only somewhat recovered to ~95% by the last iteration. Even worse, the suboptimal behavior of a wide-legged stance returned.

## Runs 036 - 038

### Hypothesis:

The learning can output near-perfect results like in the first seed, but it isn't stable enough to carry over to other seeds. The H1s did learn to survive meaningfully well in the second seed, but the wide-legged stance must be addressed. By simply increasing the joint_deviation penalty, the wide-legged stance should be overcome in any seed.

### Changes:

- Increased joint_deviation penalty

### Result:

All 3 seeds performed very well this time. At i999, they all have 99.99%+ survival rates with nearly optimal behaviors. Although it is not perfect, and two of the three seeds show signs of instability in the later (~i800-i950) iterations, this is a good enough position to deem this task officially completed. The H1s can now stand up straight on their own two legs.
