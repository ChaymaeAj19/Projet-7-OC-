import os
import joblib
import pandas as pd
import shap
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# === Chemins ===
current_directory = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline1.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "features_for_prediction.csv")

# === Vérification des fichiers requis ===
for path, label in [(pipeline_path, "pipeline complet"), (csv_path, "CSV")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {label} non trouvé : {path}")

# === Chargement du pipeline et des noms de colonnes attendues ===
model_bundle = joblib.load(pipeline_path)
pipeline = model_bundle['pipeline']
expected_features = model_bundle['features']

# Pour SHAP (modèle et explainer extraits du pipeline)
model = getattr(pipeline, 'steps', [])[::-1][0][1]  # Le dernier step est souvent le modèle
explainer = shap.TreeExplainer(model)

# === Routes ===
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data.get('SK_ID_CURR')
    with_shap = data.get('with_shap', False)

    if sk_id_curr is None:
        return jsonify({'error': "Champ 'SK_ID_CURR' requis"}), 400

    df = pd.read_csv(csv_path)

    if 'SK_ID_CURR' not in df.columns:
        return jsonify({'error': "La colonne 'SK_ID_CURR' est absente du fichier CSV"}), 500

    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    if sample.empty:
        return jsonify({'error': f"Aucun client trouvé avec SK_ID_CURR = {sk_id_curr}"}), 404

    try:
        sample_input = sample[expected_features].copy()
        sample_input = sample_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        proba = pipeline.predict_proba(sample_input)[0][1]

        result = {
            'probability': round(proba * 100, 2),
        }

        if with_shap:
            shap_values = explainer.shap_values(sample_input)
            result.update({
                'shap_values': shap_values[1][0].tolist(),
                'feature_names': sample_input.columns.tolist(),
                'feature_values': sample_input.iloc[0].tolist(),
            })
        else:
            result['message'] = "SHAP désactivé (ajouter 'with_shap': true pour l'activer)"

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Erreur pendant la prédiction : {str(e)}"}), 500

# === Lancement ===
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
