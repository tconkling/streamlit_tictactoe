# streamlit_tictactoe

Adapts https://medium.com/swlh/tic-tac-toe-and-deep-neural-networks-ea600bc53f51 to Streamlit

[s4t deployment](https://insight2020a.streamlit.io/tconkling/streamlit_tictactoe/tictactoe/)

## Issues

- `st.progress` float vs int is confusing
    - It should clamp its arguments (and maybe warn)
- Very surprised by `st.cache` dehydration time for my game simulations!
- Auto-rerunning failed _constantly_ with weird, un-reproducible errors.
    - I continue to want multiprocessing while in "data exploration" mode
- Tensorflow: so many frigging errors!
- s4t: requirements.txt changes not picked up; wish restart happened automatically
- `st.empty` seems to take a while to be applied...?