# Q-Learning for Tic Tac Toe Game

To generate Q-table in csv format for Tic Tac Toe Game run:
```python
pip install -r requirements.txt
python q_learn.py
```

Output example

| State     | A1 | A2     | A3     | A4 | A5     | A6     | A7 | A8     | A9     |
|-----------|----|--------|--------|----|--------|--------|----|--------|--------|
| bbbbbbbbb | -1 | -1     | -1     | -1 | -1     | -1     | -1 | -1     | -1     |
| xbbobbxbb | 0  | -0.920 | -0.911 | 0  | -0.910 | -0.910 | 0  | -0.910 | -0.911 |

where:<br>
`State` - state on the 3x3 board, describes by 9 chars string, where 'b' for empty cell, 'x' and 'o';<br>
`A1`, ..., `A9` - agent rewards for step to cell 1-9 for current `State`.


