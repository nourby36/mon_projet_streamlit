import streamlit as st
from prediction import predict

# Titre et description de l'application
st.title('Prédiction de type de fleur Iris')
st.write('Entrez les caractéristiques de la fleur pour prédire son type (setosa, versicolor, ou virginica).')

# Diviser en deux colonnes pour les caractéristiques
col1, col2 = st.columns(2)

# Saisie des caractéristiques
with col1:
    sepal_length = st.slider('Longueur du sépale', 4.0, 8.0, 5.0)
    sepal_width = st.slider('Largeur du sépale', 2.0, 4.5, 3.0)

with col2:
    petal_length = st.slider('Longueur du pétale', 1.0, 7.0, 4.0)
    petal_width = st.slider('Largeur du pétale', 0.1, 2.5, 1.5)

# Lorsque l'utilisateur clique sur le bouton, faire la prédiction
if st.button('Prédire'):
    features = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict(features)
    st.write(f'La prédiction du modèle est : {prediction[0]}')
