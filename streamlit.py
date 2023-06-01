import pandas as pd
import numpy as np
import sklearn as sk
import streamlit as st

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

st.title('Sturzerkennung')

# Feld für Drag&Drop fuer Testdaten

data_load_state = st.text('Loading data...')

