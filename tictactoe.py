import time

import streamlit as st

from game import *


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
        if (now - last_update) >= update_rate:
            progress_bar.progress(float(ii / count))
            last_update = now
    progress_bar.progress(1.0)


@st.cache(suppress_st_warning=True)
def simulate_games(num_games):
    notice = st.text(f"Simulating {num_games} games...")
    games = list(game for game in iterate_with_progress(num_games, simulateGame))
    notice.empty()
    return games


NUM_GAMES = 1000
games = simulate_games(NUM_GAMES)
st.write("Player 1 stats: ", gameStats(games, 1))
