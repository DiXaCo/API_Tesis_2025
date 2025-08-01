
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import pandas as pd
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import base64
import uuid

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"], supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "..", "model", "modelos_experimento_B.pkl")
LIME_DIR = os.path.join(BASE_DIR, "..", "static", "lime")
os.makedirs(LIME_DIR, exist_ok=True)

modelos = joblib.load(MODELO_PATH)
used_features = modelos['random_forest'].used_features

def obtener_resultado(modelo_nombre, modelo, datos_input):
    X_df = pd.DataFrame([datos_input])[used_features]
    pred = modelo.predict(X_df)[0]
    proba = modelo.predict_proba(X_df)[0].max()

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_df),
        feature_names=used_features,
        class_names=modelo.classes_,
        mode="classification",
        discretize_continuous=True
    )
    exp = explainer.explain_instance(
        X_df.iloc[0],
        lambda x: modelo.predict_proba(pd.DataFrame(x, columns=used_features)),
        num_features=10
    )

    unique_id = uuid.uuid4().hex[:8]
    lime_filename = f"lime_{modelo_nombre}_{unique_id}.png"
    lime_path = os.path.join(LIME_DIR, lime_filename)
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(lime_path)
    plt.close(fig)

    with open(lime_path, "rb") as image_file:
        lime_b64 = base64.b64encode(image_file.read()).decode("utf-8")

    return {
        "modelo": modelo_nombre,
        "prediccion": str(pred),
        "probabilidad": round(float(proba), 4),
        "lime": lime_b64
    }

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        modelo_seleccionado = data.get("modelo")
        entrada = data.get("entrada", {})
        # Corregir clave esperada por el modelo
        entrada["MAIN_ACTIVITY_general public\\services"] = entrada.pop("MAIN_ACTIVITY_general_public_services", 0)

        if not entrada or not modelo_seleccionado:
            return jsonify({"error": "Faltan datos de entrada o modelo"}), 400

        respuesta = {}

        if modelo_seleccionado == "random_forest":
            respuesta["random_forest"] = obtener_resultado("random_forest", modelos["random_forest"], entrada)

        elif modelo_seleccionado == "gradient_boosting":
            respuesta["gradient_boosting"] = obtener_resultado("gradient_boosting", modelos["gradient_boosting"], entrada)

        elif modelo_seleccionado == "both":
            rf_result = obtener_resultado("random_forest", modelos["random_forest"], entrada)
            gb_result = obtener_resultado("gradient_boosting", modelos["gradient_boosting"], entrada)

            respuesta["random_forest"] = rf_result
            respuesta["gradient_boosting"] = gb_result
            respuesta["mensaje"] = (
                "✅ Ambos modelos coinciden." if rf_result["prediccion"] == gb_result["prediccion"]
                else "⚠️ Los modelos difieren. Recomendamos revisar manualmente."
            )
        else:
            return jsonify({"error": "Modelo no reconocido"}), 400

        return jsonify(respuesta)

    except Exception as e:
        print("❌ Error inesperado:", e)
        return jsonify({"error": "Error interno del servidor"}), 500

if __name__ == "__main__":
    app.run(debug=True)
