import streamlit as st

st.title("Herzlich Willkommen zu FallDetector")
st.write("FallDetector hat den Anspruch Stürze zu erkennen und diese zu melden und von alltäglichen Verhaltensweisen abgrenzen zu können.")

st.title("Der Prozess zur Ideenentwicklung:")

st.write("Unsere Ideen waren vielseitig...")
if st.button("Untergrunderkennung für Fahrrad, Skateboard, E-Scooter"):
    st.write("- Ermittlung der Untergrundbeschaffenheit durch Vibration")
    st.write("- Wir fahren über eine gewisse Art Untergrund und Labeln dann die Daten") 
    st.write("- Unsere App gibt dann aus, wie schnell eine Person fahren sollte bzw. Drosselt die Geschwindigkeit")
if st.button("Risikoprofil für Autofahrer/Verkehrsteilnehmer"):
    st.write("- Risikoeinschätzung des Fahrverhaltens, Aufstellung individueller Fahrprofile")
if st.button("Herzfrequenz bei sportlichen Aktivitäten: Krankheitsbild erkennen"):
    st.write("- Marker identifizieren, die auf ein Krankheitsbild hindeuten")
if st.button("Sturzerkennung für ältere Menschen"):
    st.write("- Beschleunigungssensor, Gyroskop, Orientation und Gravity Sensoren als Ausgangspunkt für die Erkennung und Warnung von Stürzen")

st.title("Anforderungsspezifikation:")
st.markdown("- Stürze erkennen")
st.markdown("- Alltägliche Verhaltensweisen von Stürzen abgrenzen") 
st.markdown("- Stürze melden")
st.markdown("- Datenaufzeichnung sollte für jeden möglich sein, Verwendung von SensorLogger")

