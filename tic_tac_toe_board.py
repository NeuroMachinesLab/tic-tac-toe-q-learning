import numpy as np


class TicTacToeBoard:
    win_cells = [
        [0, 1, 2],  # 1-st row
        [3, 4, 5],  # 2-nd row
        [6, 7, 8],  # 3-d  row
        [0, 3, 6],  # 1-st col
        [1, 4, 7],  # 2-nd col
        [2, 5, 8],  # 3-d  col
        [0, 4, 8],  # 1-st diagonal
        [2, 4, 6],  # 2-nd diagonal
    ]
    player_cnt = 2
    action_cnt = 9
    state_cnt = pow(3, 9)  # 3 positions (X, O, empty) for 9 cells
    penalty = -1
    # Don't use lower value. It's negate earlier moves if they can lead to win.
    # Don't use higher value. This rewards the condition, leading to the opponent's winning position.
    penalty_for_opponent_pre_winning_state = penalty
    # Should be greater than penalty.
    # Don't use 0, this value in q-table used for unresolved move.
    # The reward should compensate for the entire penalty: (penalty * max_moves_cnt + reward) == 0
    reward = abs(penalty) * action_cnt

    def __init__(self):
        self.board = np.full(TicTacToeBoard.action_cnt, fill_value=-1, dtype=int)

    def reset(self):
        self.board = np.full(TicTacToeBoard.action_cnt, fill_value=-1, dtype=int)  # 3x3 board values: -1 - for empty, '0' - for 'O', 1 - for 'X'
        return self.state()

    def state(self):
        state = 0
        for i in range(9):
            state += (self.board[i] + 1) * int(pow(3, 8 - i))
        return state

    @staticmethod
    def state_name(state):
        """
        :return: board description in 9 chars string.
        String char index corresponds to cell index and value ane of 'x', 'o', '-' for blank
        """
        result = ""
        while state > 0:
            remainder = state % 3
            if remainder == 0:
                result = '-' + result
            elif remainder == 1:
                result = 'o' + result
            else:
                result = 'x' + result
            state //= 3  # Integer division
        return result.rjust(9, '-')

    def allowed_actions(self):
        return [i for i in range(9) if self.board[i] == -1]

    def do_action(self, player, action):
        """
        Sets X or O symbol on board
        :param player: 0 for 'O' player, 1 for 'X' player
        :param action: index of cell 0...8
        :return: agent reward
        """
        # check is other player wins if moves to this position
        other_player = 0 if player == 1 else 0
        self.board[action] = other_player
        possible_winner = self.who_wins()
        # do action
        self.board[action] = player
        winner = self.who_wins()
        if winner == player:
            reward = TicTacToeBoard.reward
        elif possible_winner == other_player:
            reward = TicTacToeBoard.penalty_for_opponent_pre_winning_state
        else:
            reward = TicTacToeBoard.penalty
        done = winner != -1
        return self.state(), reward, done

    def who_wins(self):
        """
        :return: -1 - no winner, 0 - 'O' player wins, 1 - 'X' player wins
        """
        for cells in TicTacToeBoard.win_cells:
            o_win = True
            x_win = True
            for cell in cells:
                v = self.board[cell]
                if v == -1:
                    o_win = False
                    x_win = False
                    break
                elif v == 1:
                    o_win = False
                elif v == 0:
                    x_win = False
            if o_win:
                return 0
            elif x_win:
                return 1
        return -1
