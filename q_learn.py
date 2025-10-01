#
# See https://habr.com/ru/companies/otus/articles/803041/
#
import time
import numpy as np
from tic_tac_toe_board import TicTacToeBoard

start_time_ns = time.time_ns()

# Learning parameters
alpha = 0.1  # learning rage
gamma = 0.99  # discount factor
epsilon = 0.1  # 10% steps are random
num_episodes = 100_000

# Обучение
env = TicTacToeBoard()

state_cnt = pow(3, 9)  # 3 positions (X, O, empty) for 9 cells
action_cnt = 9
Q = np.zeros((state_cnt, action_cnt))

player = 1
for i in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        actions = env.allowed_actions()
        if len(actions) == 0:
            break  # no one wins
        if np.random.rand() < epsilon:
            i = np.random.randint(0, len(actions))  # random action
            action = actions[i]
        else:
            # action with max Q-value
            action = actions[0]
            max_q = Q[state][action]
            for i in actions:
                q = Q[state][i]
                if q > max_q:
                    max_q = q
                    action = i

        next_state, reward, done = env.do_action(player, action)
        Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]))

        state = next_state
        player = 2 if player == 1 else 1

time_spent_s = (time.time_ns() - start_time_ns) / 1_000_000_000

print("Q-table:")
print("State, Action 1, Action 2, Action 3, Action 4, Action 5, Action 6, Action 7, Action 8, Action 9")
cnt = 0
for state in range(0, state_cnt):
    if sum(Q[state]) != 0:
        cnt += 1
        print("\n" + TicTacToeBoard.state_name(state), end="")
        for q in Q[state]:
            print(", " + f"{q:+.6f}", end="")
print("\n")
print("Total states: " + str(cnt))
print("Spent time: " + f"{time_spent_s:.3f} s")
