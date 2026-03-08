DQN Notebook
------------


I started by implementing a tabular q-learning approach. This uses a q-table, where rows are different states, and columns are different actions. Each cell is a state-action value (q-value), and it's an estimate of how good taking that action in that state would be. 

In this process, the agent learns by taking an action according to the Epsilon-Greedy Strategy, where a random number is generated and compared against epsilon to determine whether the agent takes a random action or the best possible action in the CURRENT observation state. The environment progresses with this action and the reward/state of the agent in the next action are returned. Based on this reward, as well as the current q-values for the agent at this state/action and the next state (in the next state, the maximum possible q-value is selected), the current q-value is updated. This process repeats, allowing for the q-values to better reflect the rewards for picking each state.

This is enhanced to use a neural network in the DQN approach. The purpose of this is to learn the Q function that provides the best action to take for a given observation.

A DQN uses two networks (policy network and target network). The target network updates slowly and is used to provide a concrete target for the agent to learn against. The target network eventually converges to the correct answer. Learning in this consistent way (even though the answer is initially wrong) is still effective.

A DQN also uses replay memory to sample previous experiences. This helps the network not forget prior experiences in its training.

----------------------------------------------------------------------------------------------------------------------------------
PPO Notebook
------------
PPO updates an agent's policy with policy gradients that adjust the agent in the general direction of what is "correct". It uses clipping to limit large changes between each policy adjustment. PPO also calculates advantage by comparing the actual benefit to the expected value of that state, and uses this advantage measurement to guide the policy adjustments. This expected value is a rough estimate computed from a Generalized Advantage Estimation function.