"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
"""
import numpy as np


class Reverse_utils():

    def check_if_is_inside_board(self, pos1, pos2):
        """
        Function to check coordinates if they are in the board

        :param pos1: position on axis x
        :param pos2: position on axis y
        :return: position that is on the board
        """
        return (0 <= pos1 <= 3) and (0 <= pos2 <= 3)

    def flip1(self, possiblePosition, current_player, board):
        """
        Function to check if pawn should be flipped (neighbor pawn value in given direction is different from current
        player)

        :param possiblePosition: position to set pawn
        :param current_player: represents current_player
        :param board: game board
        :return: true or false
        """
        return board[possiblePosition[0], possiblePosition[1]] == 3 - current_player

    def flip2(self, possiblePosition, current_player, board):
        """
        Function to check if last pawn from given direction has value of current player (checks if pawns checked by
        flip1 are correct to be flipped)

        :param possiblePosition: position to set pawn
        :param current_player: represents current_player
        :param board: game board
        :return: true or false
        """
        return board[possiblePosition[0], possiblePosition[1]] == current_player

    def directions(self):
        """
        Possible directions with axis x/y

        :return: list of directions to check where place a pawn
        """
        return np.array([[-1, -1],
                         [-1, 0],
                         [-1, 1],
                         [0, -1],
                         [0, 1],
                         [1, -1],
                         [1, 0],
                         [1, 1]])
