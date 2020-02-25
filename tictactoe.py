"""Adapted from https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51"""

import time

import streamlit as st
import tensorflow as tf
from tensorflow import keras

from game import model, game
from game.STProgressLogger import STProgressLogger

# Reset internal session state
tf.keras.backend.clear_session()


def iterate_with_progress(count, f):
    """Run a function a given number of times, yielding
    its result after each run. Show a progress bar.
    """
    progress_bar = st.progress(0)
    last_update = time.time()
    update_rate = 1.0 / 30.0
    for ii in range(count):
        yield f()
        now = time.time()
        if ii == count - 1:
            progress_bar.progress(1.0)
        elif (now - last_update) >= update_rate:
            progress_bar.progress(float(ii / count))
            last_update = now
    progress_bar.empty()


@st.cache(suppress_st_warning=True)
def create_training_games(count):
    notice = st.text(f"Building {count} simulated games...")
    result = list(g for g in iterate_with_progress(count, game.simulateGame))
    notice.empty()
    return result


def get_model(num_games):
    """Load or train our Keras model."""
    filename = f"model_{num_games}.h5"
    try:
        mdl = keras.models.load_model(filename)
        return mdl
    except BaseException as e:
        print(f"Failed to load {filename}: {e}")
        pass

    mdl = model.getModel()
    games = create_training_games(num_games)

    notice = st.text(f"Training model...")

    x_train, x_test, y_train, y_test = model.gamesToWinLossData(games)
    progress_logger = STProgressLogger()

    history = mdl.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=100,
        batch_size=100,
        callbacks=[
            progress_logger,
        ]
    )
    notice.empty()

    keras.models.save_model(mdl, filename)

    return mdl

# Build training data
NUM_GAMES = 10000
model_bytes = get_model(NUM_GAMES)

st.write("Done!")
