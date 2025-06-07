import random
import numpy as np
import pandas as pd

import os
import pandas as pd

# Load CSV with fallback to sample
if os.path.exists("games.csv"):
    df = pd.read_csv("games.csv")
    print("Loaded games.csv")
elif os.path.exists("sample_games.csv"):
    df = pd.read_csv("sample_games.csv")
    print("Loaded sample_games.csv")
else:
    df = pd.DataFrame()
    print("No CSV found. Using empty DataFrame.")



class ChessEnvironment:

    def __init__(self, dataset):
        self.dataset = dataset
        self.current_game = None
        self.current_move_index = 0
        self.winner = None
        self.moves = []

    def reset(self, random_game=True):
        if random_game:
            self.current_game = self.dataset.sample().iloc[0]
        self.moves = self.current_game['moves'].split()
        self.current_move_index = 0
        self.winner = self.current_game['winner']
        return self.get_state(), self.get_legal_actions()

    def get_state(self):
        return " ".join(self.moves[:self.current_move_index])

    def get_legal_actions(self):
        if self.current_move_index < len(self.moves):
            return [self.moves[self.current_move_index], random.choice(self.moves)]
        return []

    def step(self, action):
        if self.current_move_index < len(self.moves) and action == self.moves[self.current_move_index]:
            self.current_move_index += 1
            reward = 1
            finished = self.current_move_index == len(self.moves)
            return self.get_state(), self.get_legal_actions(), reward, finished
        else:
            return self.get_state(), self.get_legal_actions(), -1, True


class DoubleQLearningAgent:

    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.q_table_a = {}
        self.q_table_b = {}

    def get_q_values(self, state, table):
        if table == 'a':
            return self.q_table_a.setdefault(state, {})
        else:
            return self.q_table_b.setdefault(state, {})

    def choose_action(self, state, legal_actions):
        if np.random.rand() < self.epsilon or not legal_actions:
            return random.choice(legal_actions) if legal_actions else None

        average_q_values = {action: (self.get_q_values(state, 'a').get(action, 0) +
                                     self.get_q_values(state, 'b').get(action, 0)) / 2 for action in legal_actions}
        return max(legal_actions, key=lambda action: average_q_values.get(action, 0))

    def learn(self, state, action, reward, next_state, legal_actions, finished):
        if random.choice(['a', 'b']) == 'a':
            q_values = self.get_q_values(state, 'a')
            next_q_values_a = self.get_q_values(next_state, 'a')
            next_q_values_b = self.get_q_values(next_state, 'b')
            best_next_action = max(next_q_values_b, key=next_q_values_b.get, default=None)
            target = reward + self.discount_factor * next_q_values_a.get(best_next_action,
                                                                         0) if not finished and best_next_action else reward
            q_values[action] = q_values.get(action, 0) + self.learning_rate * (target - q_values.get(action, 0))
        else:
            q_values = self.get_q_values(state, 'b')
            next_q_values_a = self.get_q_values(next_state, 'a')
            next_q_values_b = self.get_q_values(next_state, 'b')
            best_next_action = max(next_q_values_a, key=next_q_values_a.get, default=None)
            target = reward + self.discount_factor * next_q_values_b.get(best_next_action,
                                                                         0) if not finished and best_next_action else reward
            q_values[action] = q_values.get(action, 0) + self.learning_rate * (target - q_values.get(action, 0))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def evaluate_model(env, agent, dataset, num_games=None):
    if num_games is None:
        num_games = len(dataset)
    correct_predictions = 0
    total_moves = 0

    for _ in range(num_games):
        env.reset(random_game=False)
        finished = False

        while not finished:
            state = env.get_state()
            legal_actions = env.get_legal_actions()
            if not legal_actions:
                break

            action = agent.choose_action(state, legal_actions)
            if action == env.moves[env.current_move_index]:
                correct_predictions += 1
            total_moves += 1
            _, _, _, finished = env.step(action)

    accuracy = (correct_predictions / total_moves) * 100 if total_moves > 0 else 0
    return accuracy


data_path = "games.csv"
dataset = pd.read_csv(data_path)

env = ChessEnvironment(dataset)
agent = DoubleQLearningAgent()

num_episodes = int(input("Number of games to test:"))
for episode in range(num_episodes):
    state, legal_actions = env.reset()
    finished = False
    moves_taken = []

    while not finished:
        action = agent.choose_action(state, legal_actions)
        next_state, next_legal_actions, reward, finished = env.step(action)
        agent.learn(state, action, reward, next_state, next_legal_actions, finished)
        state, legal_actions = next_state, next_legal_actions
        moves_taken.append(action)

    print(f"Game {episode + 1}: Winner = {env.winner}")
    print(f"Moves: {' '.join(filter(None, moves_taken))}")
    print("*" * 50)

accuracy = evaluate_model(env, agent, dataset, num_episodes)
print(f"Model Accuracy over {num_episodes} games: {accuracy:.2f}%")