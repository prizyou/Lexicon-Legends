import pandas as pd
import numpy as np
import sklearn as sk
import streamlit as st
from io import StringIO
import pickle as pk


from tsfresh import extract_features
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

st.title('Sturzerkennung mit Daten aus der Smartphone-App SensorLogger')

# Feld für Drag&Drop fuer Testdaten
uploaded_file = st.file_uploader("Choose a file as .JSON")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    #st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    #st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    #st.write(string_data)

    # Can be used wherever a "file-like" object is accepted:
    data = pd.read_json(uploaded_file)
    
    
    if data is not None:
        # Daten einlesen und aufbereiten
        data['time'] = pd.to_datetime(data['time'])

        data = data.set_index('time')

        data_acc = data[data['sensor'] == 'AccelerometerUncalibrated']

        data_gyro = data[data['sensor'] == 'GyroscopeUncalibrated']

        data_or = data[data['sensor'] == 'Orientation']

        data_gravity = data[data['sensor'] == 'Gravity']

        data_acc = data_acc[['z','x','y']]

        data_gyro = data_gyro[['z','x','y']]

        data_or = data_or[['qx','qz','qw','qy']]

        data_gravity = data_gravity[['z','x','y']]

        data_gyro.rename(columns={ 'z': 'gz' , 'x': 'gx' , 'y': 'gy'}, inplace=True)

        st.write("Darstellung der aufbereiteten Daten:")
        st.write("Beschleunigungssensor")
        st.line_chart(data_acc)
        st.write("Gyroskop")
        st.line_chart(data_gyro)    
        st.write("Orientierungssensor")
        st.line_chart(data_or)
        st.write("Gravitationssensor")  
        st.line_chart(data_gravity)

        #data_acc = data_acc.reset_index(inplace=True)
        #data_gyro = data_gyro.reset_index(inplace=True)

        st.write(data_acc.dtype)

        data_combine = pd.merge(data_acc, data_gyro, left_index=True, right_index=True)

        #data_combine = data_combine.reset_index(inplace=True)

        st.write(data_combine)

        #Aufteilung des Datensatzes in Sequenzen
        data_combine['id'] = 0

        id = 1

        var1 = 100

        for i in range(0, len(data_combine)):
            data_combine.iloc[i,7] = id
            
            if i >= var1: 
                var1 = var1 + 100
                id+=1

        st.write('Hochgeladene Daten:')
        st.dataframe(data_combine)

        features_filtered_direct = extract_features(data_combine,column_id='id', column_sort='time')

        st.title("Vorhersage Label in Modell:")

        #Vortrainierte Modelle laden
        model_knn = pk.load(open('knnpickle_file','rb'),)
        #model_rf = sk.models.load_model('Model_rf')

        #Schätzungsdaten rausziehen
        y_pred_knn = model_knn.predict(features_filtered_direct)
        #y_pred_rf = model_rf.predict(features_filtered_direct)

        #Vorhersage Label in Modell
        st.write("Vorhersage Label in KNN-Modell:")
        st.write(y_pred_knn)

        st.write("Vorhersage Label in RF-Modell:")
        #st.write(y_pred_rf)

