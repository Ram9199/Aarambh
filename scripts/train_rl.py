import os
import sys
import numpy as np
import random
import torch

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 

from aarambh.dialogue_environment import DialogueEnvironment
from aarambh.dqn_agent import DQNAgent

def train_dqn_agent(episodes=1000):
    env = DialogueEnvironment()
    state_size = 1  # Simplified for this example
    action_size = 3  # Three possible actions: positive, neutral, negative
    agent = DQNAgent(state_size, action_size)
    batch_size = 32

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            reward = float(reward) if not done else -10.0
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"Episode: {e}/{episodes}, Score: {time}, Epsilon: {agent.epsilon:.2}")
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        if e % 10 == 0:
            save_path = os.path.join(root_path, f"models/aarambh_dqn_{e}.pth")
            agent.save(save_path)

if __name__ == "__main__":
    os.makedirs(os.path.join(root_path, 'models'), exist_ok=True)
    train_dqn_agent()
