import shap  # Ajout de l'import SHAP
import numpy as np

# Initialisation du modèle global SHAP (à faire une seule fois)
model = pipeline.named_steps['classifier']  # Adapter ce nom selon votre pipeline
explainer = shap.TreeExplainer(model)

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

        sample_input = sample_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Calcul de la probabilité
        proba = pipeline.predict_proba(sample_input)[0][1]

        # Calcul des valeurs SHAP
        # Attention : on doit appliquer le même prétraitement que dans le pipeline
        transformed_input = pipeline.named_steps['preprocessor'].transform(sample_input)
        shap_values = explainer.shap_values(transformed_input)

        # Assurez-vous que shap_values[1] correspond à la classe positive
        shap_dict = dict(zip(expected_features, shap_values[1][0]))

        # On trie les features les plus influentes
        sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)

        return jsonify({
            'probability': round(proba * 100, 2),
            'shap_values': sorted_shap
        })

    except Exception as e:
        return jsonify({'error': f"Erreur pendant la prédiction : {str(e)}"}), 500
