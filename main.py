import streamlit as st
import pandas as pd
import numpy as np

st.title('Fahrverhalten: Routen zur Universität')

DATE_COLUMN = 'time'
DATA = ('bike_ride.json')

data_load_state = st.text('Loading data...')

df_example = pd.read_json(DATA)

#preprocessing data
df_example['time'] = pd.to_datetime(df_example['time'])
df_example = df_example.set_index('time')
df_example['sensor'].unique()


df_example_acc = df_example[df_example['sensor'] == 'AccelerometerUncalibrated']

st.write(df_example_acc.head(5))

if st.checkbox('Include x,y Data'):
    st.subheader('Raw data')
    df_example_acc = df_example_acc[['z','x','y']]
    st.line_chart(data=df_example_acc)


st.subheader('Aufgezeichnete Bewegungen über die Fahrt')
st.write('z-Achse Ausschlag vertikal, x,y Ausschlag horizontal')

df_example_acc = df_example_acc['z']

st.line_chart(data=df_example_acc)

