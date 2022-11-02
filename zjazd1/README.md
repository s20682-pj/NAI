
# Reversi (easier version)

Authors: 
Zuzanna Ciborowska s20682 & Joanna Walkiewicz s20161

Here is an easier version of the Reversi game. The game is played on a 4x4 board.
The rows are numbered with the letters A-D, and the columns with the numbers 1-4.
To place a pawn, you need to write the selected box, e.g.: D2.

Each pawn played must be laid adjacent to an opponent's pawn so that the opponent's pawn
or a row of opponent's pawn is flanked by the new pawn and another pawn of the player's colour.
All of the opponent's pawns between these two pawns are 'captured' and turned over to match the player's colour.

It can happen that a pawn is played so that pawns or rows of pawns
in more than one direction are trapped between the new pawn played and other pawns of the same colour.
In this case, all the pawns in all viable directions are turned over.

The game is over when neither player has a legal move (i.e. a move that captures at least one opposing pawn)
or when the board is full.

Instructions how to play full version of Reversi can be find here: https://www.mastersofgames.com/rules/reversi-othello-rules.htm

To run the game please install:
- Python 3.10 (https://www.python.org/downloads/)
- pip (curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py ; python get-pip.py)
- EasyAI (pip install easyAI)
- Numpy (pip install numpy)
