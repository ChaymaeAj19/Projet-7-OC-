from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- Chargement modèle et données brutes ---
model_path = os.path.join("Simulations", "Best_model", "lgbm_pipeline1.pkl")
df_path = os.path.join("Simulations", "Data", "features_for_prediction.csv")

pipeline_bundle = joblib.load(model_path)
pipeline = pipeline_bundle["pipeline"]
expected_features = pipeline_bundle["features"]

df_global = pd.read_csv(df_path)  # données brutes, non encodées

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    sk_id_curr = data.get("SK_ID_CURR")
    data_modified = data.get("data", None)  # Liste avec dict des données modifiées

    try:
        if data_modified is not None:
            # Cas modification : construire DataFrame à partir des données modifiées
            df_input = pd.DataFrame(data_modified)

            # Vérifier que toutes les colonnes attendues sont présentes
            missing_cols = set(expected_features) - set(df_input.columns)
            if missing_cols:
                return jsonify({"error": f"Colonnes manquantes: {missing_cols}"}), 400

            # Prédiction avec pipeline complet (inclut encodage, scaling, etc.)
            proba = pipeline.predict_proba(df_input[expected_features])[0][1]

        else:
            # Cas prédiction classique : extraire ligne dans df_global
            sample = df_global[df_global["SK_ID_CURR"] == sk_id_curr]
            if sample.empty:
                return jsonify({"error": "SK_ID_CURR inconnu"}), 404

            proba = pipeline.predict_proba(sample[expected_features])[0][1]

        return jsonify({"SK_ID_CURR": sk_id_curr, "probability": float(proba * 100)})

    except Exception as e:
        return jsonify({"error": f"Erreur traitement données: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
