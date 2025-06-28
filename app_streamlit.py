import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

st.set_page_config(layout="wide")

# === Chargement du modèle et des données ===
model_path = os.path.join("Simulations", "Best_model", "lgbm_pipeline1.pkl")
data_path = os.path.join("Simulations", "Data", "features_for_prediction.csv")

# Vérifications
if not os.path.exists(model_path) or not os.path.exists(data_path):
    st.error("Modèle ou données manquantes. Vérifiez les chemins.")
    st.stop()

# Chargement
model_bundle = joblib.load(model_path)
pipeline = model_bundle['pipeline']
expected_features = model_bundle['features']
model = pipeline.steps[-1][1]  # Modèle LightGBM (LGBMClassifier)

# Données clients
data = pd.read_csv(data_path)
if 'SK_ID_CURR' not in data.columns:
    st.error("Colonne 'SK_ID_CURR' manquante dans les données.")
    st.stop()

# Données filtrées
X_all = data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

# === Création de l'explainer SHAP (global) ===
if hasattr(model, "booster_"):
    booster = model.booster_
else:
    booster = model

explainer = shap.TreeExplainer(booster)

# === Sidebar ===
st.sidebar.header("Sélection du client")
client_id = st.sidebar.selectbox("Choisir un identifiant client :", data['SK_ID_CURR'].unique())

# === Titre ===
st.title("Dashboard de Prédiction Crédit Client")
st.write("Analyse du risque de défaut et explication par SHAP")

# === Affichage client ===
client_data = data[data["SK_ID_CURR"] == client_id]
X_client = client_data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

# === Info client ===
st.subheader("Informations client")
st.dataframe(client_data)

# === Prédiction ===
st.subheader("Probabilité de défaut")
proba = pipeline.predict_proba(X_client)[0][1]
st.write(f"**Probabilité de défaut :** {proba:.2%}")

# Décision
seuil = 0.5
decision = "✅ Prêt accordé" if proba < seuil else "❌ Prêt refusé"
st.markdown(f"### Décision : <span style='color:{'green' if proba < seuil else 'red'}'>{decision}</span>", unsafe_allow_html=True)

# === SHAP local ===
st.subheader("Explication locale SHAP")

# Obtenir l'explication SHAP
explanation = explainer(X_client)

# Sélectionner la sortie classe 1 si nécessaire (pour classifier)
if hasattr(explanation[0], "shape") and len(explanation[0].base_values.shape) > 0:
    explanation_local = explanation[0].select(output=1)
else:
    explanation_local = explanation[0]

fig, ax = plt.subplots()
shap.plots.waterfall(explanation_local, show=False)
st.pyplot(fig)

# === SHAP global ===
st.subheader("Explication globale SHAP (features les plus importantes)")

global_shap_vals = explainer.shap_values(X_all)
shap_val_global = global_shap_vals[1] if isinstance(global_shap_vals, list) else global_shap_vals

fig2, ax2 = plt.subplots()
shap.summary_plot(shap_val_global, X_all, plot_type='bar', show=False, max_display=10)
st.pyplot(fig2)
