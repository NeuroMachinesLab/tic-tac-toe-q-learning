#
# See https://habr.com/ru/companies/otus/articles/803041/
#
import time
import numpy as np
from tic_tac_toe_board import TicTacToeBoard as board

start_time_ns = time.time_ns()

# Learning parameters
num_episodes = 100_0000
epsilon = 1.0  # 0 - the world is not studied, 1 - studying moves (each action receives equal chances for estimation)
#avg_num_episodes_per_state = float(num_episodes) / board.state_cnt
#alpha = 1.0 / avg_num_episodes_per_state  # learning rage (use 1/N value, where each state episode makes an equal contribution)
#gamma = 1.0  # discount factor, 0 - for ignoring future benefits, 1 - for future result estimation

# Learning
Q = np.zeros((board.player_cnt, board.state_cnt, board.action_cnt), dtype=np.longdouble)
env = board()

for episode in range(num_episodes):
    if episode % 100_000 == 0: print(f'Learning ({episode:,} episodes done)')
    # The greater episode value, the more accurate Q table, so the prediction is more accurate.
    # Train network with greater learning rate (alpha) and discount factor (gamma)
    alpha = min(0.9, episode / num_episodes)
    gamma = alpha

    state = env.reset()
    player = 1
    done = False
    while not done:
        # Select next action
        actions = env.allowed_actions()
        if len(actions) == 0:
            break  # no one wins
        if np.random.rand() < epsilon:
            i = np.random.randint(0, len(actions))  # random action, studying move
            action = actions[i]
        else:
            # Action with max Q-value, the world is not studied
            # Start with random action for case when all actions have same q-value
            action = actions[np.random.randint(0, len(actions))]
            max_q = Q[player][state][action]
            for i in actions:
                q = Q[player][state][i]
                if q > max_q:
                    max_q = q
                    action = i

        # Update Q table
        next_state, reward, done = env.do_action(player, action)
        # Finds max(Q(next_state))
        # Don't use max_Q_next_state = 0. It is incorrect for case when all other values is negative
        max_Q_next_state = float("-inf")
        next_player = 0 if player == 1 else 1
        for q in Q[next_player][next_state]:
            if q > max_Q_next_state and q != 0.0:  # 0.0 is for unknown value, don't use it
                max_Q_next_state = q
        if max_Q_next_state == float("-inf"):
            max_Q_next_state = 0  # max(Q(next_state)) is unknown, use 0
        # Use negative sign for max(Q(next_state)), because:
        # the benefits of the next player are the penalty of the current player
        Q[player][state][action] = (1 - alpha) * Q[player][state][action] + alpha * (reward + gamma * -max_Q_next_state)

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
