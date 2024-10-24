import joblib
import numpy as np

# Fonction pour charger le modèle
def load_model():
    model = joblib.load('C:/Users/MSI/Desktop/IDS5/MLOps/mon_projet_streamlit/model.pkl')
    return model

# Fonction pour faire des prédictions
def predict(features):
    model = load_model()
    return model.predict([features])
