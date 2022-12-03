# CS551 Final Project
This is a Streamlit-based machine learning web app made to predict the pH of a body of water given its other chemical elements
- Training and app file were written in Python
- The model was trained using the tpot and sklearn libraries.
- The app was made using the streamlit library and hosted via Streamlit as well
Dependences can be found in the requirements.txt.

The web app can be accessed at: https://davidwli1-cs551-final-app-d0klu4.streamlit.app/
When opening the app for the first time, please be patient during the time it takes for Streamlit to boot it up.
Also, it may take a few seconds for the entirety of the app to load as it must make queries to Wikipedia first.

Text files and assets used in the web app can be found in the "app_contents" and "pictures" subdirectories, respectively.
The final trained model can be accessed in the "final_model.joblib" file, and its Python code in the "pipeline.py" file.
The raw data used in the training Python file can be found in the "data" subdirectory.
Data used by my classmates to help test the web app can be found in the "peer data" subdirectory.

Originally this was going to be an app about predicting the density of zooplankton in water instead of pH.
The original (uncommented, and probably unworking) Python file used is in "notebook (old).ipynb", and the data it uses can be found in the "old data" subdirectory"
