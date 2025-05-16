from flask import Flask, jsonify, request, render_template
import os
import joblib
import pandas as pd
import shap

app = Flask(__name__)

# === Paths ===
current_directory = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline.pkl")
preprocessor_path = os.path.join(current_directory, "Simulations", "Scaler", "StandardScaler1.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train_sample.csv")

# === Vérification des fichiers ===
for path, label in [(model_path, "modèle"), (preprocessor_path, "préprocesseur"), (csv_path, "CSV")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier {label} non trouvé : {path}")

# === Chargement des objets ===
model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

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
    sample = df[df['SK_ID_CURR'] == sk_id_curr]
    if sample.empty:
        return jsonify({'error': f"Aucun client trouvé avec SK_ID_CURR = {sk_id_curr}"}), 404

    # Séparer SK_ID_CURR
    sample_input = sample.drop(columns=['SK_ID_CURR'])

    # Transformation des données (encodage + imputation + scaling)
    sample_processed = preprocessor.transform(sample_input)

    # Prédiction
    proba = model.predict_proba(sample_processed)[0][1]

    # SHAP explainer (TreeExplainer compatible avec LightGBM ou XGBoost)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_processed)

    return jsonify({
        'probability': round(proba * 100, 2),
        'shap_values': shap_values[1][0].tolist(),
        'feature_names': sample_input.columns.tolist(),
        'feature_values': sample_input.values[0].tolist()
    })

# === Lancement de l'application ===
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
