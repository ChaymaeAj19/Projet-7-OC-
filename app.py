import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request

app = Flask(__name__)

# Répertoire courant du script
current_directory = os.path.abspath(os.path.dirname(__file__))

# Chemins vers les fichiers
model_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline.pkl")
scaler_path = os.path.join(current_directory, "Simulations", "Scaler", "StandardScaler.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train.csv")

# Vérifications de l'existence des fichiers
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Fichier modèle non trouvé : {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Fichier scaler non trouvé : {scaler_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Fichier CSV non trouvé : {csv_path}")

# Chargement du modèle et scaler
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data.get('SK_ID_CURR')
    if sk_id_curr is None:
        return jsonify({'error': "Champ 'SK_ID_CURR' requis"}), 400

    # Charger le CSV
    df = pd.read_csv(csv_path)

    # Filtrer l'échantillon
    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    if sample.empty:
        return jsonify({'error': f"Aucun client trouvé avec SK_ID_CURR = {sk_id_curr}"}), 404

    # Supprimer la colonne ID pour la prédiction
    sample = sample.drop(columns=['SK_ID_CURR'])

    # Appliquer le scaler
    sample_scaled = scaler.transform(sample)

    # Prédire
    prediction = model.predict_proba(sample_scaled)
    proba = prediction[0][1]  # Probabilité classe positive

    # Valeurs SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_scaled)

    return jsonify({
        'probability': round(proba * 100, 2),
        'shap_values': shap_values[1][0].tolist(),
        'feature_names': sample.columns.tolist(),
        'feature_values': sample.values[0].tolist()
    })

if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
