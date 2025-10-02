#
# See https://habr.com/ru/companies/otus/articles/803041/
#
import time
import numpy as np
from tic_tac_toe_board import TicTacToeBoard

start_time_ns = time.time_ns()

# Learning parameters
alpha = 0.1  # learning rage
gamma = 0.99 # discount factor
epsilon = 0.1  # 10% steps are random
num_episodes = 100_000

# Learning
state_cnt = pow(3, 9)  # 3 positions (X, O, empty) for 9 cells
action_cnt = 9
Q = np.zeros((state_cnt, action_cnt))

env = TicTacToeBoard()
for episode in range(num_episodes):
    if episode % 100_000 == 0: print(f'Learning ({episode:,} episodes done)')
    state = env.reset()
    player = 1
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
            action = actions[np.random.randint(0, len(actions))]  # if all actions have same value select random
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
print(f'Learning ({num_episodes:,} episodes done)')

# Write to file
print()
print("Writing to 'q-table.csv'")
cnt = 0
with open('q-table.csv', 'w', newline='') as file:
    file.write(
        'State,      Action 1,  Action 2,  Action 3,  Action 4,  Action 5,  Action 6,  Action 7,  Action 8,  Action 9')
    for state in range(0, state_cnt):
        if sum(Q[state]) != 0:
            cnt += 1
            file.write('\n')
            file.write(TicTacToeBoard.state_name(state))
            for q in Q[state]:
                file.write(', ')
                file.write(f'{q:+.6f}')

print('Total states: ' + str(cnt))
print('Spent time: ' + f'{time_spent_s:.3f} s')
