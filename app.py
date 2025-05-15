from flask import Flask, jsonify, request, render_template
import os
import joblib
import pandas as pd
import shap

app = Flask(__name__)

current_directory = os.path.abspath(os.path.dirname(__file__))

model_path = os.path.join(current_directory, "Simulations", "Best_model", "lgbm_pipeline.pkl")
scaler_path = os.path.join(current_directory, "Simulations", "Scaler", "StandardScaler.pkl")
csv_path = os.path.join(current_directory, "Simulations", "Data", "df_train.csv")

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Fichier modèle non trouvé : {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Fichier scaler non trouvé : {scaler_path}")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Fichier CSV non trouvé : {csv_path}")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

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

    sample = sample.drop(columns=['SK_ID_CURR'])
    sample_scaled = scaler.transform(sample)

    prediction = model.predict_proba(sample_scaled)
    proba = prediction[0][1]

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
