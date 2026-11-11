# Architecture notes

The app is a single-process Streamlit deployment. `streamlit_app.py` is the only entry point and imports the `cleansense` package using a path anchored to the project root, avoiding ambiguous `app.py` imports and nested working-directory assumptions.

The engine writes uploaded originals and cleaned outputs to local storage. CSV profiling uses pandas chunk iteration for row and missing-value aggregates. CSV duplicate removal uses a temporary SQLite primary-key table to detect duplicates across chunk boundaries. Profiling distributions, sensitive-data signals, and model comparison are sample-based and visibly labelled. Source files are not overwritten.
