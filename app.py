import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
from streamlit_lottie import st_lottie
import json
import time

# --- CONFIG DE LA PAGE ---
st.set_page_config(page_title="Wine Quality Predictor 🍷", page_icon="🍷", layout="centered")

# --- FOND D'ÉCRAN AVEC BASE64 ---
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                         url("data:image/jpeg;base64,{encoded}");
             background-size: cover;
             background-position: center;
             background-attachment: fixed;
         }}

         /* Conteneur principal : semi-transparent + bien lisible */
         section.main {{
             background-color: rgba(20, 20, 20, 0.85);
             padding: 2rem;
             border-radius: 15px;
         }}

         h1, h2, p, label, .stButton>button {{
             color: #F5F5F5;
         }}

         .stButton>button {{
             background-color: #800000;
             color: white;
             border-radius: 10px;
         }}

         .stButton>button:hover {{
             background-color: #A52A2A;
         }}
         </style>
         """,
         unsafe_allow_html=True
    )

# --- APPEL DU BACKGROUND ---
add_bg_from_local("vignoble.jpeg")

# --- CHARGEMENT DE L'ANIMATION LOTTIE ---
def load_lottie_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

animation = load_lottie_file("wine_animation.json")  # Mets ton fichier Lottie ici

# --- CHARGEMENT DES OBJETS ---
model = joblib.load("model.joblib")
#scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("features.joblib")

# --- MAP DE SMILEYS PAR CARACTÉRISTIQUE ---
feature_emojis = {
    "fixed_acidity": "🍇",
    "volatile_acidity": "💨",
    "citric_acid": "🍋",
    "residual_sugar": "🍬",
    "chlorides": "🧂",
    "free_sulfur_dioxide": "💨",
    "total_sulfur_dioxide": "⚗️",
    "density": "⚖️",
    "pH": "💧",
    "sulphates": "🥂",
    "alcohol": "🍷"
}

# --- TITRE ---
st.markdown("<h1 style='text-align: center;'>🍷 Wine Quality Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Remplis les caractéristiques chimiques du vin pour obtenir une prédiction de sa qualité (note de 1 à 9)</p>", unsafe_allow_html=True)
st.divider()

# --- SAISIE UTILISATEUR ---
user_input = {}
col1, col2 = st.columns(2)
for idx, feature in enumerate(feature_names):
    col = col1 if idx % 2 == 0 else col2
    emoji = feature_emojis.get(feature, "🍇")  # Emoji par défaut si pas trouvé
    with col:
        user_input[feature] = st.number_input(
            label=f"{emoji} {feature.replace('_', ' ').capitalize()}",
            min_value=0.0,
            max_value=450.0,
            value=5.0,
            step=0.1
        )

# --- PREDICTION ---
st.divider()
if st.button("🔮 Prédire la qualité du vin", use_container_width=True):
    with st.spinner("Analyse en cours... 🍇🔍"):
        time.sleep(1.5)  # Simulation d'une attente avant la prédiction
        input_df = pd.DataFrame([user_input])
        #input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_df)[0]
        prediction_rounded = int(round(prediction))

        st_lottie(animation, height=300, key="wine_anim")  # Affichage de l'animation Lottie

        # Message stylisé avec la prédiction
        st.markdown(
        f"<h2 style='text-align: center; color: #FFD700;'>✨ Note estimée : {prediction:.2f} (≈ {prediction_rounded}/9)</h2>",
        unsafe_allow_html=True
    )

# Afficher les probabilités pour chaque classe
#if hasattr(model, "predict_proba"):
#    proba = model.predict_proba(scaled_data)[0]
#    st.bar_chart(proba)
