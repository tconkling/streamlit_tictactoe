"""Basic tictactoe data & logic"""

import random

import pandas as pd


def initBoard():
    """Create an empty board."""
    board = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    return board


def printBoard(board):
    """Get the state of the board as a string."""
    out = ""
    for i in range(len(board)):
        for j in range(len(board[i])):
            mark = ' '
            if board[i][j] == 1:
                mark = 'X'
            elif board[i][j] == 2:
                mark = 'O'

            out += mark
            if j == len(board[i]) - 1:
                out += "\n"
            else:
                out += "|"
        if i < len(board) - 1:
            out += "-----\n"

    return out


def getMoves(board):
    """Get a list of valid moves (indices into the board)"""
    moves = []
    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == 0:
                moves.append((i, j))
    return moves


def getWinnerString(result):
    if result == -1:
        return "Game not over"
    elif result == 0:
        return "Draw"
    elif result == 1:
        return "Player 1 (X)"
    elif result == 2:
        return "Player 2 (O)"
    else:
        return f"Invalid result {result}"


# Declare a winner
#
# -1 = game not over
#  0 = draw
#  1 = 'X' wins (player 1)
#  2 = 'O' wins (player 2)
def getWinner(board):
    candidate = 0
    won = 0

    # Check rows
    for i in range(len(board)):
        candidate = 0
        for j in range(len(board[i])):

            # Make sure there are no gaps
            if board[i][j] == 0:
                break

            # Identify the front-runner
            if candidate == 0:
                candidate = board[i][j]

            # Determine whether the front-runner has all the slots
            if candidate != board[i][j]:
                break
            elif j == len(board[i]) - 1:
                won = candidate

    if won > 0:
        return won

    # Check columns
    for j in range(len(board[0])):
        candidate = 0
        for i in range(len(board)):

            # Make sure there are no gaps
            if board[i][j] == 0:
                break

            # Identify the front-runner
            if candidate == 0:
                candidate = board[i][j]

            # Determine whether the front-runner has all the slots
            if candidate != board[i][j]:
                break
            elif i == len(board) - 1:
                won = candidate

    if won > 0:
        return won

    # Check diagonals
    candidate = 0
    for i in range(len(board)):
        if board[i][i] == 0:
            break
        if candidate == 0:
            candidate = board[i][i]
        if candidate != board[i][i]:
            break
        elif i == len(board) - 1:
            won = candidate

    if won > 0:
        return won

    candidate = 0
    for i in range(len(board)):
        if board[2 - i][2 - i] == 0:
            break
        if candidate == 0:
            candidate = board[2 - i][2 - i]
        if candidate != board[2 - i][2 - i]:
            break
        elif i == len(board) - 1:
            won = candidate

    if won > 0:
        return won

    # Still no winner?
    if (len(getMoves(board)) == 0):
        # It's a draw
        return 0
    else:
        # Still more moves to make
        return -1


def movesToBoard(moves):
    """Construct a board from a move history"""
    board = initBoard()
    for move in moves:
        player = move[0]
        coords = move[1]
        board[coords[0]][coords[1]] = player
    return board


def gameStats(games, player=1):
    """Return a dataframe with stats from the games"""
    stats = {"win": 0, "loss": 0, "draw": 0}
    for game in games:
        result = getWinner(movesToBoard(game))
        if result == -1:
            continue
        elif result == player:
            stats["win"] += 1
        elif result == 0:
            stats["draw"] += 1
        else:
            stats["loss"] += 1

    return pd.DataFrame({
        "Wins": [stats["win"] / len(games) * 100],
        "Loss": [stats["loss"] / len(games) * 100],
        "Draw": [stats["draw"] / len(games) * 100],
    })
