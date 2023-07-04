Sturzerkennung mittels der App SensorLogger:

Sturzerkennung mit FaLLDetector:copyright: 
Ein Projekt von: Anitan, Paul, Max

Das Projekt soll es ermöglichen, Stürze mit Hilfe von Smartphones zu erkennen. Dazu werden die Sensordaten des Smartphones aufgezeichnet und in einem Machine Learning Modell analysiert. Das Modell kann dann in einer App implementiert werden, um Stürze zu erkennen und Hilfe zu rufen.")
Die App ist in Python geschrieben und nutzt die Bibliotheken Streamlit, Pandas, Numpy, Scikit-Learn und Tsfresh.
Diese verwendet als Datenquelle SensorLogger aus dem AppStore:

Es werden Sensordaten des Smartphones aufgezeichnet und diese in einer .JSON Datei abgespeichert. Die App kann so eingestellt werden, dass sie nur die relevanten Sensoren aufzeichnet:
Accelerometer, Gyroscope, Orientation, Gravity


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

https://jneuroengrehab.biomedcentral.com/articles/10.1186/s12984-021-00918-z"""
