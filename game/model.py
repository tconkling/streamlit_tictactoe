"""Model for training and playing tictactoe"""

import numpy as np
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential

from game.game import *


def getModel():
    """Create a Keras model for learning tic-tac-toe."""
    numCells = 9
    outcomes = 3
    model = Sequential()
    model.add(Dense(200, activation='relu', input_shape=(numCells, )))
    model.add(Dropout(0.2))
    model.add(Dense(125, activation='relu'))
    model.add(Dense(75, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(25, activation='relu'))
    model.add(Dense(outcomes, activation='softmax'))
    model.compile(
        loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc']
    )
    return model


def gamesToWinLossData(games):
    """Transform games histories into training data for our model.
    """
    x = []
    y = []
    for game in games:
        winner = getWinner(movesToBoard(game))
        for move in range(len(game)):
            x.append(movesToBoard(game[:(move + 1)]))
            y.append(winner)

    x = np.array(x).reshape((-1, 9))
    y = to_categorical(y)

    # Return an appropriate train/test split
    trainNum = int(len(x) * 0.8)
    return x[:trainNum], x[trainNum:], y[:trainNum], y[trainNum:]


def simulateGame(p1=None, p2=None, rnd=0):
    """Simulate a complete game and return the move history."""
    history = []
    board = initBoard()
    playerToMove = 1

    while getWinner(board) == -1:
        # Chose a move (random or use a player model if provided)
        move = None
        if playerToMove == 1 and p1 != None:
            move = bestMove(board, p1, playerToMove, rnd)
        elif playerToMove == 2 and p2 != None:
            move = bestMove(board, p2, playerToMove, rnd)
        else:
            moves = getMoves(board)
            move = moves[random.randint(0, len(moves) - 1)]

        # Make the move
        board[move[0]][move[1]] = playerToMove

        # Add the move to the history
        history.append((playerToMove, move))

        # Switch the active player
        playerToMove = 1 if playerToMove == 2 else 2

    return history


def bestMove(board, model, player, rnd=0):
    """Choose the move with the best predicted outcome"""
    scores = []
    moves = getMoves(board)

    # Make predictions for each possible move
    for i in range(len(moves)):
        future = np.array(board)
        future[moves[i][0]][moves[i][1]] = player
        prediction = model.predict(future.reshape((-1, 9)))[0]
        if player == 1:
            winPrediction = prediction[1]
            lossPrediction = prediction[2]
        else:
            winPrediction = prediction[2]
            lossPrediction = prediction[1]
        drawPrediction = prediction[0]
        if winPrediction - lossPrediction > 0:
            scores.append(winPrediction - lossPrediction)
        else:
            scores.append(drawPrediction - lossPrediction)

    # Choose the best move with a random factor
    bestMoves = np.flip(np.argsort(scores))
    for i in range(len(bestMoves)):
        if random.random() * rnd < 0.5:
            return moves[bestMoves[i]]

    # Choose a move completely at random
    return moves[random.randint(0, len(moves) - 1)]
