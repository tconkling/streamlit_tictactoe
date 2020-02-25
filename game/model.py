"""Model for training and playing tictactoe"""

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from keras.utils.np_utils import to_categorical

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
