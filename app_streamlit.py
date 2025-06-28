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
model = pipeline.steps[-1][1]  # Modèle LightGBM

# Données clients
data = pd.read_csv(data_path)
if 'SK_ID_CURR' not in data.columns:
    st.error("Colonne 'SK_ID_CURR' manquante dans les données.")
    st.stop()

# Données filtrées
X_all = data[expected_features].copy().apply(pd.to_numeric, errors='coerce').fillna(0)

# === Création de l'explainer SHAP ===
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

# Obtenir les shap_values
shap_values_local_all = explainer.shap_values(X_client)

# Pour les classifieurs binaires, shap_values est une liste
if isinstance(shap_values_local_all, list) and len(shap_values_local_all) == 2:
    shap_values_local = shap_values_local_all[1][0]  # Classe 1, 1er client
    base_value = explainer.expected_value[1]
else:
    shap_values_local = shap_values_local_all[0]
    base_value = explainer.expected_value

# Construction de l'objet SHAP Explanation
explanation_local = shap.Explanation(
    values=shap_values_local,
    base_values=base_value,
    data=X_client.iloc[0],
    feature_names=X_client.columns.tolist()
)

fig, ax = plt.subplots()
shap.plots.waterfall(explanation_local, show=False)
st.pyplot(fig)

# === SHAP global ===
st.subheader("Explication globale SHAP (features les plus importantes)")

shap_values_global_all = explainer.shap_values(X_all)
shap_values_global = shap_values_global_all[1] if isinstance(shap_values_global_all, list) else shap_values_global_all

fig2, ax2 = plt.subplots()
shap.summary_plot(shap_values_global, X_all, plot_type='bar', show=False, max_display=10)
st.pyplot(fig2)
