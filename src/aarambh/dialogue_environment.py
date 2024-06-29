import random
import numpy as np

class DialogueEnvironment:
    def __init__(self):
        self.state = None
        self.reset()
        
    def reset(self):
        # Reset the environment to an initial state
        self.state = "initial"
        return self._encode_state(self.state)
    
    def step(self, action):
        # Execute the action and return the new state, reward, and done flag
        user_response = self.simulate_user_response(action)
        reward = self.calculate_reward(action, user_response)
        done = self.is_done(user_response)
        self.state = user_response
        return self._encode_state(self.state), reward, done
    
    def simulate_user_response(self, action):
        # Simulate a user response based on the agent's action
        # For simplicity, we use random responses
        possible_responses = ["positive", "neutral", "negative"]
        return random.choice(possible_responses)
    
    def calculate_reward(self, action, user_response):
        # Define the reward system
        if user_response == "positive":
            return 1  # Positive reward for a good response
        elif user_response == "neutral":
            return 0  # No reward for a neutral response
        else:
            return -1  # Negative reward for a bad response
    
    def is_done(self, user_response):
        # Define the condition to end the conversation
        return user_response == "negative"
    
    def _encode_state(self, state):
        # Convert the state to a numerical format
        state_dict = {"initial": 0, "positive": 1, "neutral": 2, "negative": 3}
        return np.array([state_dict[state]], dtype=np.float32)
