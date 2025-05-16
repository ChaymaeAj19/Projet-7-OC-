from flask import Flask, jsonify, request, render_template
import os
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# === Paths ===
current_directory = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "features_for_prediction.csv")

# === Vérification des fichiers ===
for path, label in [(pipeline_path, "pipeline complet"), (csv_path, "CSV")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {label} non trouvé : {path}")

# === Chargement du pipeline complet ===
pipeline = joblib.load(pipeline_path)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data.get('SK_ID_CURR')
    if sk_id_curr is None:
        return jsonify({'error': "Champ 'SK_ID_CURR' requis"}), 400

    # Charger les données
    df = pd.read_csv(csv_path)
    if 'SK_ID_CURR' not in df.columns:
        return jsonify({'error': "La colonne 'SK_ID_CURR' est absente du fichier CSV"}), 500

    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    if sample.empty:
        return jsonify({'error': f"Aucun client trouvé avec SK_ID_CURR = {sk_id_curr}"}), 404

    # Préparation des données (exclure SK_ID_CURR)
    sample_input = sample.drop(columns=['SK_ID_CURR'])

    # S'assurer que les colonnes sont dans le bon ordre et qu'elles correspondent
    try:
        expected_features = pipeline.named_steps['model'].feature_name_
        sample_input = sample_input[expected_features]
    except Exception as e:
        return jsonify({'error': f"Erreur dans le réarrangement des colonnes : {str(e)}"}), 500

    try:
        # Prédiction avec pipeline complet
        proba = pipeline.predict_proba(sample_input)[0][1]

        # SHAP values
        model = pipeline.named_steps['model']
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_input)

        return jsonify({
            'probability': round(proba * 100, 2),
            'shap_values': shap_values[1][0].tolist(),  # Classe 1
            'feature_names': sample_input.columns.tolist(),
            'feature_values': sample_input.values[0].tolist()
        })

    except Exception as e:
        return jsonify({'error': f"Erreur pendant la prédiction : {str(e)}"}), 500

# === Lancement de l'application ===
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
