from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pandas as pd
import joblib
from Util.Util import generar_explica_lime
from Util.reglas_lime import convert_to_if_then

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"], supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "model", "modelos_experimento_B.pkl")
modelos = joblib.load(MODELO_PATH)

used_features = modelos['random_forest'].used_features
clases = modelos['random_forest'].classes_

@app.route("/", methods=["GET"])
def index():
    return render_template("formulario.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        modelo_nombre = data.get("modelo")
        entrada = data.get("entrada", {})

        entrada["MAIN_ACTIVITY_general public\\services"] = entrada.pop("MAIN_ACTIVITY_general_public_services", 0)

        for feature in used_features:
            if feature not in entrada:
                entrada[feature] = 0

        df = pd.DataFrame([entrada])[used_features]
        modelo = modelos.get(modelo_nombre)

        if modelo is None:
            return jsonify({"error": "Modelo no reconocido"}), 400

        pred = modelo.predict(df)
        proba = modelo.predict_proba(df)[0].max()
        lime_img = generar_explica_lime(modelo, df, used_features)

        # Wrapper para que LIME pase un DataFrame a predict_proba
        def predict_proba_wrapper(x_array):
            df_temp = pd.DataFrame(x_array, columns=used_features)
            return modelo.predict_proba(df_temp)

        from lime.lime_tabular import LimeTabularExplainer
        explainer = LimeTabularExplainer(
            training_data=df.values,
            feature_names=df.columns,
            class_names=modelo.classes_,
            mode='classification'
        )
        exp = explainer.explain_instance(
            df.iloc[0],
            predict_proba_wrapper,
            num_features=len(df.columns)
        )

        reglas_texto = {}
        for clase in exp.available_labels():
            reglas_texto[str(clase)] = convert_to_if_then(exp, clase, pred[0])

        return jsonify({
            "prediccion": str(pred[0]),
            "probabilidad": round(float(proba), 4),
            "explicacion_lime": lime_img,
            "reglas_por_clase": reglas_texto
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno", "mensaje": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

    