import time

import streamlit as st
from keras.callbacks import ProgbarLogger

from game import model, game
from game.STProgressLogger import STProgressLogger


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
def simulate_games(num_games):
    notice = st.text(f"Simulating {num_games} games...")
    games = list(g for g in iterate_with_progress(num_games, game.simulateGame))
    notice.empty()
    return games


NUM_GAMES = 1000
games = simulate_games(NUM_GAMES)

mdl = model.getModel()
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
