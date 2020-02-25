"""Adapted from https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51"""
import os
import time

import streamlit as st
import tensorflow as tf
from tensorflow import keras

from game import model, game
from game.STProgressLogger import STProgressLogger

# Reset internal session state
tf.keras.backend.clear_session()


def iterate_with_progress(f, count, notice_text=None):
    """Run a function a given number of times, yielding
    its result after each run. Show a progress bar.
    """
    notice = None
    if notice_text is not None:
        notice = st.text(notice_text)

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
    if notice is not None:
        notice.text("")
        notice.empty()


@st.cache(suppress_st_warning=True)
def create_training_games(count):
    return list(g for g in iterate_with_progress(
        model.simulateGame,
        count,
        f"Building {count} simulated games..."
    ))


def get_model(num_games):
    """Load or train our Keras model."""
    filename = os.path.abspath(f"./model_{num_games}.h5")
    notice = st.text(f"Loading {filename}...")
    try:
        mdl = keras.models.load_model(filename)
        notice.text(f"Loading {filename}... Done!")
        return mdl
    except BaseException as e:
        print(f"Failed to load {filename}: {e}")
        pass

    notice.text(f"Training model ({num_games} games)...")
    mdl = model.getModel()
    if num_games > 0:
        games = create_training_games(num_games)

        x_train, x_test, y_train, y_test = model.gamesToWinLossData(games)
        progress_logger = STProgressLogger()

        history = mdl.fit(
            x=x_train,
            y=y_train,
            validation_data=(x_test, y_test),
            epochs=100,
            batch_size=100,
            verbose=0,
            callbacks=[
                progress_logger,
            ]
        )

    notice.text(f"Training model ({num_games} games)... Done!")
    keras.models.save_model(mdl, filename)

    return mdl

st.image("shall_we_play.jpg")

# Build training data
num_training_games = st.number_input(
    label="Num training games",
    min_value=0,
    value=500,
    step=100
)
mdl = get_model(num_training_games)

# Simulate
num_played_games = st.number_input(
    label="Num played games",
    min_value=0,
    value=50,
    step=20,
)

if st.button("Play!"):
    played_games = list(g for g in iterate_with_progress(
        lambda: model.simulateGame(p1=mdl),
        num_played_games,
        f"Playing {num_played_games} games..."
    ))

    st.write(game.gameStats(played_games))

