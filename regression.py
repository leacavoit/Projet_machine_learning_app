import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Chargement du dataset Wine Quality (rouge + blanc)
wine_quality = fetch_ucirepo(id=186)
X = wine_quality.data.features
y = wine_quality.data.targets.squeeze()

# 2. Séparation des données
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Mise à l'échelle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# 4. Entraînement du modèle final
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Sauvegarde du modèle et du scaler
joblib.dump(model, "model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(X_train.columns.tolist(), "features.joblib")


print("✅ Modèle et scaler sauvegardés avec succès !")
