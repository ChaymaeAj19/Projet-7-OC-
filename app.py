import os
import io
import base64
import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
import gc
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

# === Chargement du pipeline et des colonnes ===
model_bundle = joblib.load(pipeline_path)
pipeline = model_bundle['pipeline']
expected_features = model_bundle['features']

# === Chargement global du CSV pour éviter les relectures répétées ===
df_global = pd.read_csv(csv_path)

# === Extraction du modèle pour SHAP ===
model = pipeline.steps[-1][1]
explainer = shap.TreeExplainer(model)

# === ROUTE Accueil ===
@app.route("/")
def index():
    return render_template("index.html")

# === ROUTE Prédiction personnalisée ===
@app.route("/predict", methods=['POST'])
def predict():
    data = request.json
    sk_id_curr = data.get('SK_ID_CURR')
    with_shap = data.get('with_shap', False)

    if sk_id_curr is None:
        return jsonify({'error': "Champ 'SK_ID_CURR' requis"}), 400

    if 'SK_ID_CURR' not in df_global.columns:
        return jsonify({'error': "La colonne 'SK_ID_CURR' est absente du fichier CSV"}), 500

    sample = df_global[df_global['SK_ID_CURR'] == sk_id_curr]
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

            # === SHAP local (waterfall) ===
            shap.initjs()
            shap_value = shap_values[1][0]
            feature_names = sample_input.columns

            explanation = shap.Explanation(
                values=shap_value,
                base_values=explainer.expected_value[1],
                data=sample_input.iloc[0],
                feature_names=feature_names
            )

            plt.clf()
            shap.plots._waterfall.waterfall_legacy(explanation, show=False)

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            buf.close()

            result.update({
                'shap_plot_base64': img_base64
            })

            gc.collect()

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f"Erreur pendant la prédiction : {str(e)}"}), 500

# === ROUTE SHAP Global ===
@app.route("/shap_global")
def shap_global():
    try:
        # Échantillonnage aléatoire de 50 lignes pour limiter la mémoire
        data_input = df_global[expected_features].sample(n=50, random_state=42).copy()
        data_input = data_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        shap_values = explainer.shap_values(data_input)

        plt.clf()
        shap.summary_plot(
            shap_values[1],
            data_input,
            plot_type="bar",
            show=False,
            max_display=10
        )

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        gc.collect()

        return jsonify({"image": img_base64})

    except Exception as e:
        return jsonify({"error": f"Erreur lors du calcul du SHAP global : {str(e)}"}), 500

# === Lancement ===
if __name__ == "__main__":
    port = os.environ.get("PORT", 5000)
    app.run(debug=False, host="0.0.0.0", port=int(port))
