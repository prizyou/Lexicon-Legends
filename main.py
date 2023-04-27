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

st.subheader('Vibrations per Ride')

df_example_acc = df_example_acc['z']

st.line_chart(data=df_example_acc)
#fig = df_example_acc.plot(figsize=(10,5))
#st.show(fig)
