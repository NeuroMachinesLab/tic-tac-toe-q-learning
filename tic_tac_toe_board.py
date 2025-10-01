import numpy as np


class TicTacToeBoard:
    win_cells = [
        [0, 1, 2],  # 1-st row
        [3, 4, 5],  # 2-nd row
        [6, 7, 8],  # 3-d  row
        [0, 2, 6],  # 1-st col
        [1, 4, 7],  # 2-nd col
        [2, 5, 8],  # 3-d  col
        [0, 4, 8],  # 1-st diagonal
        [2, 4, 6],  # 2-nd diagonal
    ]

    def reset(self):
        self.board = np.zeros(9).astype(int)  # 3x3 board values: 0 - for empty, 1 - for 'X', '2' - for 'O'
        return self.state()

    def state(self):
        state = 0
        for i in range(9):
            state += self.board[i] * int(pow(3, 8 - i))
        return state

    @staticmethod
    def state_name(state):
        """
        :return: board description in 9 chars string.
        String char index corresponds to cell index and value ane of 'X', 'O', 'B' for blank
        """
        result = ""
        while state > 0:
            remainder = state % 3
            if remainder == 0:
                result = 'b' + result
            elif remainder == 1:
                result = 'x' + result
            else:
                result = 'o' + result
            state //= 3  # Integer division
        return result.rjust(9, 'b')

    def allowed_actions(self):
        return [i for i in range(9) if self.board[i] == 0]

    def do_action(self, player, action):
        """
        Sets X or O symbol on board
        :param player: 1 for 'X' player, 2 for 'O' player
        :param action: index of cell 0...8
        :return: agent reward
        """
        self.board[action] = player
        winner = self.who_wins()
        reward = 0 if winner == player else -1
        done = winner != 0
        return self.state(), reward, done

    def who_wins(self):
        """
        :return: 0 - no winner, 1 - 'X' player wins, 2 - 'O' player wins
        """
        for cells in TicTacToeBoard.win_cells:
            result = self.sum_not_empty_cells(cells)
            if result == 3:
                return 1
            elif result == 6:
                return 2
        return 0

    def sum_not_empty_cells(self, cells):
        sum = 0
        for i in cells:
            c = self.board[i]
            if c == 0:
                return -1  # cell is empty
            else:
                sum += c
        return sum
