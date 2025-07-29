from flask import Flask, request, jsonify
import joblib
import pandas as pd
from pathlib import Path

app = Flask(__name__)


#Ruta
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "PROCESO_MODELADO.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET"])
def home():
    return "✅ API del modelo de sugerencia está activa."

# 2. Endpoint para recibir datos y predecir
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Obtener datos del cuerpo de la petición
        input_json = request.get_json()
        data = input_json.get("data", None)

        if data is None or not isinstance(data, list):
            return jsonify({"error": "Formato incorrecto. Se esperaba una lista de listas en el campo 'data'."}), 400

        # Convertir a DataFrame con las columnas que espera el modelo
        df = pd.DataFrame(data, columns=model.used_features)

        # Hacer la predicción
        prediction = model.predict(df)

        # Devolver como respuesta
        return jsonify({
            "prediction": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
