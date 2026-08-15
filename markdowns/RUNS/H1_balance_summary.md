Teaching H1s to balance is a tougher task than it sounds, especially when going into it with limited RL knowledge. The most important concepts I have learned that relate to this task could be summed up in the following:

1. The reward landscape is where a task is made or broken. Without the right combination of rewards and penalties, the task cannot be learned. The rewards are meant to _define_ the task.

2. The observation space is just as important. Missing observations also make or break an agent's success. All of the rewards in the reward landscape need appropriate observation terms, and some observations are needed solely for the purpose of the agent's environmental understanding.

3. Custom rewards aren't always necessary. The built-in mdp rewards are simplistic, but they work well enough to teach an agent on their own. Custom rewards may muddy the training by offering little to no guidance, especially if they are implemented improperly.

4. Curriculums aren't always necessary either. For simple tasks like balance, a curriculum does little to influence or benefit the overall learning. Poorly implemented curricula can even hurt the training, such as a reward weight curriculum suddenly advancing on an already steadily improving agent.

5. It is VERY important to verify the logical integrity of your environment before all else. Regardless of if you are experimenting, just learning, or actively pursuing a task, an environment that is set up with logical flaws will steal your time unapologetically.
