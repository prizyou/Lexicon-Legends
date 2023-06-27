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

st.write("Das Projekt soll es ermöglichen, Stürze mit Hilfe von Smartphones zu erkennen. Dazu werden die Sensordaten des Smartphones aufgezeichnet und in einem Machine Learning Modell analysiert. Das Modell kann dann in einer App implementiert werden, um Stürze zu erkennen und Hilfe zu rufen.")
st.write("Die App ist in Python geschrieben und nutzt die Bibliotheken Streamlit, Pandas, Numpy, Scikit-Learn und Tsfresh.")
st.write("Diese verwendet als Datenquelle SensorLogger aus dem AppStore:")

col1, col2, col3 = st.beta_columns([1,1,1])
title_container = st.beta_container()

with title_container:
    with col1:
        st.image("pictures/1.PNG", caption='SensorLogger App',width=250,output_format="auto")
    with col2:
        st.image("pictures/2.PNG", caption='Relevant Sensors',width=250,output_format="auto")       
    with col3:
        st.image("pictures/3.PNG", caption='Settings in SensorLogger',width=250,output_format="auto")   

st.write("Es werden Sensordaten des Smartphones aufgezeichnet und diese in einer .JSON Datei abgespeichert. Die App kann so eingestellt werden, dass sie nur die relevanten Sensoren aufzeichnet:")
st.write("Accelerometer, Gyroscope, Orientation, Gravity")

st.title("Applikation:")

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

        data_gravity = data[data['sensor'] == 'Gravity']

        st.write("Die folgenden Sensoren wurden aufgezeichnet:")
        
        sensoren = data['sensor'].unique()
        sensoren = pd.DataFrame(sensoren)
        sensoren.columns = ['Sensoren: ']

        st.write(sensoren)

        data_acc = data_acc[['z','x','y']]

        data_gyro = data_gyro[['z','x','y']]

        data_or = data_or[['qx','qz','qw','qy']]

        data_gravity = data_gravity[['z','x','y']]

        data_gyro.rename(columns={ 'z': 'gz' , 'x': 'gx' , 'y': 'gy'}, inplace=True)
        data_gravity.rename(columns={ 'z': 'grav_z' , 'x': 'grav_x' , 'y': 'grav_y'}, inplace=True)

        st.write("Darstellung der aufbereiteten Daten:")
        if st.button('Beschleunigungssensor'):
            st.line_chart(data_acc)
        if st.button('Gyroskop'):
            st.line_chart(data_gyro)    
        if st.button('Orientation'):        
            st.line_chart(data_or)
        if st.button('Gravity'):
            st.line_chart(data_gravity)


        data_acc['index'] = 0
        data_gyro['index'] = 0
        data_or['index'] = 0
        data_gravity['index'] = 0

        for i in range(len(data_acc)):
            data_acc.iloc[i,3] = i
            data_gyro.iloc[i,3] = i
            data_or.iloc[i,4] = i
            data_gravity.iloc[i,3] = i        

        data_combine = pd.merge(data_acc,data_gyro, how="inner",on="index")
        data_combine = pd.merge(data_combine,data_or, how="inner",on="index")
        data_combine = pd.merge(data_combine,data_gravity, how="inner",on="index")

        #data_combine = data_combine.drop(columns=['index'])

        st.write(data_combine)

        #Aufteilung des Datensatzes in Sequenzen
        data_combine['id'] = 0

        id = 1

        var1 = 100

        for i in range(0, len(data_combine)):
            data_combine.iloc[i,14] = id
            
            if i >= var1: 
                var1 = var1 + 100
                id+=1

        
        st.header("Vorhersage der Labels in den Modellen:")

        st.write("Sequenzen der übertragenen Aufzeichnungen:")

        if st.button('Sequenzen als Dataframe'):
            st.write(data_combine)


        graph_daten = data_combine.drop(columns=['id'])

        data_combine = data_combine.reset_index(inplace=False)

        features = extract_features(data_combine,column_id='id', column_sort='index')

        while(features is None):
            st.warning("Features werden extrahiert...")    

        st.success("Features extrahiert! :)")
 
        st.success("Analyse der Daten mit KNN- und RandomForest Modell...")

    
        #Vortrainierte Modelle laden
        model_knn = pk.load(open('knnpickle_file','rb'),)
        model_rf = pk.load(open('rfpickle_file','rb'),)
        featuresList = pk.load(open('featuresList_file','rb'),)
        
        my_array = np.asarray(featuresList)
        
        if st.button('Features als Liste'):
            st.write(my_array)

        #Schätzungsdaten rausziehen
        y_pred_knn = model_knn.predict(features[[my_array[0]]])
        y_pred_rf = model_rf.predict(features[[my_array[0]]])

        st.caption("Plot der Rohdaten:")
        st.line_chart(graph_daten)   

        #Vorhersage Label in Modell
        st.header("Vorhersage Labels in KNN-Modell:")
        st.caption("In "+ str(y_pred_knn.sum()) + " der übertragenen " + str(data_combine['id'].unique().max()) + " Sequenzen aus der Aufzeichnung liegt vermutlich ein Sturz vor")
        
        y_pred_knn = pd.DataFrame(y_pred_knn)
        y_pred_knn.replace(to_replace=0, value="Normal", inplace=True)
        y_pred_knn.replace(to_replace=1, value="Fall", inplace=True)
        y_pred_knn.index += 1
        y_pred_knn.columns = ['Prediction:  ']
        st.write(y_pred_knn.T)

        st.header("Vorhersage Labels in RF-Modell:")
        st.caption("In "+ str(y_pred_rf.sum()) + " der übertragenen " + str(data_combine['id'].unique().max()) + " Sequenzen aus der Aufzeichnung liegt vermutlich ein Sturz vor")
        
        y_pred_rf = pd.DataFrame(y_pred_rf)
        y_pred_rf.replace(to_replace=0, value="Normal", inplace=True)
        y_pred_rf.replace(to_replace=1, value="Fall",inplace=True)
        y_pred_rf.index += 1
        y_pred_rf.columns = ['Prediction:  ']
        st.write(y_pred_rf.T)


st.title("Theorie:")
st.write("""Sturzerkennung mit Machine Learning Modellen: 

=> Das k-NN und RF-Modell als Modelle, welche nach den wissenschaftlichen Papern / State of the Art am besten dazu geeignet sind, Stürze zu erkennen
    
 Aktivitäts Labels: 
 
- Stehen
- Gehen
- Treppe hochlaufen (UP)
- Treppe herunterlaufen (DOWN)
- Sitzen
- Sich ins Bett legen

Wissenscahftliche Arbeiten/ Referenzen:

https://www.sciencedirect.com/science/article/pii/S1877050918318398

https://kth.diva-portal.org/smash/get/diva2:1230962/FULLTEXT01.pdf

https://www.researchgate.net/publication/308467199_Fall_Detection_Using_Machine_Learning_Algorithms

https://www.researchgate.net/publication/353576862_Latest_Research_Trends_in_Fall_Detection_and_Prevention_Using_Machine_Learning_A_Systematic_Review

https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-021-00918-z""")