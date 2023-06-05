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

st.set_page_config(
    page_title="Sturzerkennung",
    page_icon=":runner:",
    layout="wide",
)

st.title("Sturzerkennung mit FaLLDetector:copyright:  :mag:")
st.caption("Ein Projekt von: Anitan, Paul, Max")

st.sidebar.success("Klicke durch die Dokumentation...")

# Feld für Drag&Drop fuer Testdaten
uploaded_file = st.file_uploader("Laden Sie eine Datei als .JSON File hoch, um diese auf Stürze zu analysieren:")
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

        data_gravity = data[data['sensor'] == 'GravityUncalibrated']

        st.write("Die folgenden Sensoren wurden aufgezeichnet:")
        
        st.write(data['sensor'].unique())

        data_acc = data_acc[['z','x','y']]

        data_gyro = data_gyro[['z','x','y']]

        data_or = data_or[['qx','qz','qw','qy']]

        data_gravity = data_gravity[['z','x','y']]

        data_gyro.rename(columns={ 'z': 'gz' , 'x': 'gx' , 'y': 'gy'}, inplace=True)

        st.write("Darstellung der aufbereiteten Daten:")
        if st.button('Beschleunigungssensor'):
            st.line_chart(data_acc)
        if st.button('Gyroskop'):
            st.line_chart(data_gyro)    
        if st.button('Orientation'):        
            st.line_chart(data_or)
        if st.button('Gravity'):
            st.line_chart(data_gravity)

        #data_acc = data_acc.reset_index(inplace=True)
        #data_gyro = data_gyro.reset_index(inplace=True)

        data_combine = pd.merge(data_acc, data_gyro, left_index=True, right_index=True)

        #st.write(data_combine)

        #Aufteilung des Datensatzes in Sequenzen
        data_combine['id'] = 0

        id = 1

        var1 = 100

        for i in range(0, len(data_combine)):
            data_combine.iloc[i,6] = id
            
            if i >= var1: 
                var1 = var1 + 100
                id+=1

        if st.button('Daten als Dataframe'):
            st.dataframe(data_combine)

        st.title("Vorhersage der Labels in den Modellen:")

        st.write("Sequenzen der übertragenen Aufzeichnungen:")

        if st.button('Sequenzen als Dataframe'):
            st.write(data_combine)

        if st.button('Sequenzen als Grafik'):
            st.line_chart(data_combine)    

        data_combine = data_combine.reset_index(inplace=False)
        
        features = extract_features(data_combine,column_id='id', column_sort='time')
        
        while(features is None):
            st.warning("Features werden extrahiert...")    

        st.success("Features extrahiert! :)")
        
        if st.button('Features als DataFrame'):
            st.write(features)

        st.success("Analyse der Daten mit KNN- und RandomForest Modell...")
        #Vortrainierte Modelle laden
        model_knn = pk.load(open('knnpickle_file','rb'),)
        model_rf = pk.load(open('rfpickle_file','rb'),)
        featuresList = pk.load(open('featuresList_file','rb'),)
        
        #st.write(featuresList)
        my_array = np.asarray(featuresList)
        
        if st.button('Features als Liste'):
            st.write(my_array[0])
        #st.write(type(my_array))

        #Schätzungsdaten rausziehen
        y_pred_knn = model_knn.predict(features[my_array[0]])
        y_pred_rf = model_rf.predict(features[my_array[0]])
        
        y_pred_knn.replace(to_replace="0", value="Kein Sturz", inplace=True,to_replace="1", value="Sturz", inplace=True)
        y_pred_rf.replace(to_replace="0", value="Kein Sturz", inplace=True,to_replace="1", value="Sturz", inplace=True)

        #Vorhersage Label in Modell
        st.write("Vorhersage Label in KNN-Modell:")
        st.write(y_pred_knn)
        st.caption("In "+ str(y_pred_knn.sum()) + " der übertragenen " + str(data_combine['id'].unique().max()) + " Sequenzen aus der Aufzeichnung liegt vermutlich ein Sturz vor")

        st.write("Vorhersage Label in RF-Modell:")
        st.write(y_pred_rf)
        st.caption("In "+ str(y_pred_rf.sum()) + " der übertragenen " + str(data_combine['id'].unique().max()) + " Sequenzen aus der Aufzeichnung liegt vermutlich ein Sturz vor")