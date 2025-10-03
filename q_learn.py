#
# See https://habr.com/ru/companies/otus/articles/803041/
#
import time
import numpy as np
from tic_tac_toe_board import TicTacToeBoard as board

start_time_ns = time.time_ns()

# Learning parameters
num_episodes = 100_000
avg_num_episodes_per_state = float(num_episodes) / board.state_cnt
alpha = 1 / avg_num_episodes_per_state  # learning rage (use 1/N value, where each state episode makes an equal contribution)
gamma = 1  # discount factor, 0 - for ignoring future benefits, 1 - for future result estimation
epsilon = 1  # 0 - the world is not studied, 1 - studying moves (each action receives equal chances for estimation)

# Learning
Q = [[[0.0 for _ in range(board.action_cnt)] for _ in range(board.state_cnt)] for _ in range(board.player_cnt)]
env = board()

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
            max_q = Q[player][state][action]
            for i in actions:
                q = Q[player][state][i]
                if q > max_q:
                    max_q = q
                    action = i

        next_state, reward, done = env.do_action(player, action)
        next_player = 0 if player == 1 else 1
        # -max(Q[next_player][next_state]) - negative sign is used, because:
        # the benefits of the next player are the penalty of the current player
        Q[player][state][action] = ((1 - alpha) * Q[player][state][action] +
                                    alpha * (reward + gamma * -max(Q[next_player][next_state])))

        state = next_state
        player = next_player

time_spent_s = (time.time_ns() - start_time_ns) / 1_000_000_000
print(f'Learning ({num_episodes:,} episodes done)')

# Write to file
for player in range(2):
    cnt = 0
    player_name = 'o' if player == 0 else 'x'
    file_name = f'q-table-{player_name}.csv'
    with open(file_name, 'w', newline='') as file:
        file.write(
            'State,        Action 1,  Action 2,  Action 3,  Action 4,  Action 5,  Action 6,  Action 7,  Action 8,  Action 9')
        for state in range(0, board.state_cnt):
            if sum(Q[player][state]) != 0:
                cnt += 1
                file.write('\n')
                name = board.state_name(state)
                name = name[:3] + ' ' + name[3:6] + ' ' + name[6:]
                file.write(name)
                for q in Q[player][state]:
                    file.write(', ')
                    q_str = '    -    ' if q == 0.0 else f'{q:-.6f}'
                    file.write(q_str.rjust(9))
    print(f"\nTotal states for '{player_name}' player: " + str(cnt))
    print(f"Written to '{file_name}'")

print('\nSpent time: ' + f'{time_spent_s:.3f} s')
