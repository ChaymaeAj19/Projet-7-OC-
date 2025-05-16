from flask import Flask, jsonify, request, render_template
import os
import joblib
import pandas as pd

app = Flask(__name__)

# === Paths ===
current_directory = os.path.abspath(os.path.dirname(__file__))
pipeline_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline1.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "features_for_prediction.csv")

# === Vérification des fichiers ===
for path, label in [(pipeline_path, "pipeline complet"), (csv_path, "CSV")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {label} non trouvé : {path}")

# === Chargement du pipeline + noms de colonnes ===
model_bundle = joblib.load(pipeline_path)
pipeline = model_bundle['pipeline']
expected_features = model_bundle['features']

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

    df = pd.read_csv(csv_path)

    if 'SK_ID_CURR' not in df.columns:
        return jsonify({'error': "La colonne 'SK_ID_CURR' est absente du fichier CSV"}), 500

    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    if sample.empty:
        return jsonify({'error': f"Aucun client trouvé avec SK_ID_CURR = {sk_id_curr}"}), 404

    try:
        sample_input = sample[expected_features].copy()

        # Convertir tous les champs en numérique, remplacer les erreurs par NaN, puis remplacer les NaN par 0
        sample_input = sample_input.apply(pd.to_numeric, errors='coerce')
        sample_input = sample_input.fillna(0)

        proba = pipeline.predict_proba(sample_input)[0][1]

        return jsonify({
            'probability': round(proba * 100, 2),
            'message': "SHAP désactivé pour éviter les crashs mémoire"
        })

    except Exception as e:
        return jsonify({'error': f"Erreur pendant la prédiction : {str(e)}"}), 500

# === Lancement ===
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
