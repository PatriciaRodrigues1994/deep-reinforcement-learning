import numpy as np
from collections import defaultdict

class Agent:

	def __init__(self, nA=6):
		""" Initialize agent.

		Params
		======
		- nA: number of actions available to the agent
		"""
		self.nA = nA
		self.Q = defaultdict(lambda: np.zeros(self.nA))
		
		# self.gamma = 1.0

	def select_action(self, env, state, epsilon):
		""" Given the state, select an action.

		Params
		======
		- state: the current state of the environment

		Returns
		=======
		- action: an integer, compatible with the task's action space

		"""
		action = np.random.choice(np.arange(self.nA), p=self.get_probs(self.Q[state], epsilon)) if state in self.Q else env.action_space.sample()
		return action


	def get_probs(self, Q_s, epsilon):
		""" obtains the action probabilities corresponding to epsilon-greedy policy """
		policy_s = np.ones(self.nA) * epsilon / self.nA
		best_a = np.argmax(Q_s)
		policy_s[best_a] = 1 - epsilon + (epsilon / self.nA)
		return policy_s

	def step(self, state, action, reward, next_state, done, gamma, alpha, epsilon):
		""" Update the agent's knowledge, using the most recently sampled tuple.

		Params
		======
		- state: the previous state of the environment
		- action: the agent's previous choice of action
		- reward: last reward received
		- next_state: the current state of the environment
		- done: whether the episode is complete (True or False)
		"""
		self.Q[state][action] = self.Q[state][action] + (alpha * (reward + (gamma * np.dot(self.Q[next_state], self.get_probs(self.Q[next_state], epsilon))) - self.Q[state][action]) )