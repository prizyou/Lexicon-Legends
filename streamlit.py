import pandas as pd
import numpy as np
import sklearn as sk
import streamlit as st
from io import StringIO

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

st.title('Sturzerkennung')

# Feld für Drag&Drop fuer Testdaten
uploaded_file = st.file_uploader("Choose a file as .JSON")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    dataframe = pd.read_json(uploaded_file)
    st.write(dataframe)

data_load_state = st.text('Loading data...')

# Daten einlesen und aufbereiten
st.write('Hochgeladene Daten:')

st.dataframe(dataframe.style.highlight_max(axis=0))