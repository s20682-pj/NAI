"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
"""
import numpy as np
from easyAI import TwoPlayerGame
from reverse import Reverse


def to_string(a):
    """
    Convert array of board coords [x,y] to string
    eg. [1,1] to 'B2'
    :param a - array [x,y]
    :return string format of board coords
    """
    return "ABCD"[a[0]] + str(a[1] + 1)


def to_array(s):
    """
    Convert board coords in string format to array [x,y]
    eg. 'B2' to [1,1]
    :param s - string format of board cords eg. 'B1'
    :return array of board cords [x,y]
    """
    return np.array(["ABCD".index(s[0]), int(s[1]) - 1])


class Reversi(TwoPlayerGame):

    def __init__(self, players):
        """
        Initializing the game logic, board and current player
        :param any players: Human player and ai player
        """
        self.reverse = Reverse()
        self.players = players
        self.board = np.zeros((4, 4), dtype=int)
        self.board[1, 1] = 1
        self.board[1, 2] = 2
        self.board[2, 1] = 2
        self.board[2, 2] = 1
        self.current_player = 1

    def possible_moves(self):
        """
        Check if any pawn will be flipped, if yes - move is possible and it's added to list of possible moves
        :return: list of moves player can do
        """
        list_of_moves = []

        for i in range(4):
            for j in range(4):
                if self.board[i, j] == 0 and self.reverse.get(self.board, (i, j), self.current_player) != []:
                    list_of_moves.append(to_string((i, j)))

        return list_of_moves

    def make_move(self, move):
        """
        Putting the pawn on the board and flipping opponent pawns
        :param any move: coordinates of chosen place on board to put pawn
        :return: pawns on board has been reversed and new pawn has been placed
        """
        move = to_array(move)
        flipped = self.reverse.get(self.board, move, self.current_player)
        for pawn in flipped:
            self.board[pawn[0], pawn[1]] = self.current_player
        self.board[move[0], move[1]] = self.current_player

    def make_middle_board(self, i):
        """
        Filling board with pawns and '.' for empty spaces
        :param any i - row number:
        :return rows of printable characters for board
        """
        middle_board_place = [[".", "1", "2"][self.board[i][j]] for j in range(4)]
        return middle_board_place

    def show(self):
        """
        Print the board
        :return: show the board on the screen
        """
        print(
            "\n"
            + "\n".join(
                ["  1 2 3 4"]
                + [
                    "ABCD"[i]
                    + " "
                    + " ".join(self.make_middle_board(i))
                    for i in range(4)
                ]
                + [""]
            )
        )

    def is_over(self):
        """
        Conditions when the game will be finished
        :return: list of possible moves
        """
        return self.possible_moves() == []

    def scoring(self):
        """
        Counting score for the players by summing amount of pawns on the board
        :return: total score (difference of player's score and computer's score)
        """
        score_player = np.sum(self.board == self.current_player)
        score_opponent = np.sum(self.board == self.opponent_index)
        return score_player - score_opponent
