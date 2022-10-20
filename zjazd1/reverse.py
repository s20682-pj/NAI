"""
Authors: Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161
"""
from reverse_utils import Reverse_utils
import numpy as np


class Reverse():

    def __init__(self):
        """
        Connect to reverse_utils.py
        """
        self.utils = Reverse_utils()

    def get(self, board, position, current_player):
        """
        Going through the board to check which pawns should be flipped
        :param board: game board
        :param position: coordinates of chosen place on board to put pawn
        :param any current_player: index of the player whose move is
        :return: list of pawns to be reversed
        """
        possible_directions = self.utils.directions()

        reversed_pawns = []

        for direction in possible_directions:
            possible_position = position + direction
            self.to_flip(reversed_pawns, possible_position, board, current_player, direction)

        return reversed_pawns

    def to_flip(self, reversed_pawns, possible_position, board, current_player, direction):
        """
        Cheking if pawn could be flipped
        :param any reversed_pawns: reversed pawn
        :param possible_position: coordinates of chosen place on board to put pawn changed by direction
        :param board: board of game
        :param any current_player: index of the player whose move is
        :param any direction: coordinates by how much the position on the board will change
        """
        pawns_to_reverse = []
        while self.utils.check_if_is_inside_board(possible_position[0], possible_position[1]):
            if self.utils.flip1(possible_position, current_player, board):
                pawns_to_reverse.append(+possible_position)
            elif self.utils.flip2(possible_position, current_player, board):
                reversed_pawns += pawns_to_reverse
                break
            else:
                break
            possible_position += direction
