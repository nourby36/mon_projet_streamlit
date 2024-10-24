import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Charger le dataset Iris
data = pd.read_csv('data/iris.csv', header=None)
X = data.iloc[:, :-1]  # Caractéristiques (features)
y = data.iloc[:, -1]   # Label (classe de la fleur)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Évaluer la précision du modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Précision : {accuracy * 100:.2f}%')

# Sauvegarder le modèle
joblib.dump(model, 'model.pkl')
