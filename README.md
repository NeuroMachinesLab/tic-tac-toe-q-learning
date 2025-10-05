# Q-Learning for Tic Tac Toe Game

To generate Q-table in csv format for Tic Tac Toe Game run:

```python
pip install - r requirements.txt
python q_learn.py
```

Q-table output example:

| State       | A1 | A2   | A3   | A4   | A5   | A6 | A7 | A8 | A9   |
|-------------|----|------|------|------|------|----|----|----|------|
| o-- --x ox- | -  | -9.1 | -9.1 | -7.4 | -9.1 | -  | -  | -  | -9.1 |
| xbbobbxbb   | -  | -9.1 | -9.1 | -    | 7.1  | -  | -  | -  | -9.1 

where:<br>
`State` - state on the 3x3 board, describes by 9 chars string, where 'b' for empty cell, 'x' and 'o';<br>
`A1`, ..., `A9` - agent rewards for move to cell 1-9 for current `State`, higher value - the action is preferable.

The actions on gameboard are:
```
| 1 | 2 | 3
-----------
| 4 | 5 | 6
-----------
| 7 | 8 | 9
```

2 files are generated for convenience: `q-table-x.csv` and `q-table-o.csv` for X-player and O-player moves.
They may be joined, because files contains unique states.

## Details
The core of the algorithm is a Bellman equation:
```
Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward + gamma * max(Q[next_state]))
```
where:<br>
`alpha` - learning rate, determines to what extent newly acquired information overrides old information (0..1);<br>
`gamma` - discount factor, determines the importance of future rewards (0..1);<br>
`reward` - reword for move, which changes `state` to `next_state`.

This algorithm is tuned.

#### The more training, the more accurately Q-table.
Tuned algorithm doesn't use constant `alpha` and `gamma`. They calculated by:
```
alpha = gamma = i / N 
```
where `i` - the training iteration, `N` - the total training cycles.

#### Minus sign is used for `max(Q[next_state])`
The formula is corrected to
```
Q[state][action] = (1 - alpha) * Q[state][action] + alpha * (reward + gamma * -max(Q[next_state]))
```
because the next state belongs to opponent's move. But the benefits of the opponent are the penalty of the current player.
The penalty is a negative value.

#### Higher rewards
In general `penalty = -1` and `reward = 0` are used. Tuned algorithm uses:
```
reward = max_step_count * penalty 
```
where `max_step_count = 9`.

Q-table is better distinguishes the winning, if reward is propagated to the first move of the party without blur.
Nine moves with `-1` penalty is compensated by reward.
