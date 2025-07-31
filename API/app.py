from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import io
import base64
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Inicializar API Flask
app = Flask(__name__)
CORS(app)

# Cargar modelos entrenados
MODELO_PATH = Path(__file__).resolve().parent.parent / "model" / "modelos_experimento_B.pkl"
modelos = joblib.load(MODELO_PATH)
rf_model = modelos["random_forest"]
gb_model = modelos["gradient_boosting"]
used_features = rf_model.used_features

# Funciones auxiliares para gráficos
def generar_matriz_confusion(y_true, y_pred, titulo):
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(set(y_true)))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(titulo)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def generar_grafico_barras(probabilidades, titulo, clases):
    fig, ax = plt.subplots()
    ax.bar(clases, probabilidades[0])
    ax.set_title(titulo)
    ax.set_ylabel("Probabilidad")
    plt.xticks(rotation=45)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

@app.route("/", methods=["GET"])
def index():
    return "\u2705 API Experimento B activa"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.json.get("data")
        df = pd.DataFrame(input_data, columns=used_features)

        # Predicciones
        pred_rf = rf_model.predict(df).tolist()
        pred_gb = gb_model.predict(df).tolist()

        # Probabilidades y visualización (una sola fila)
        prob_rf = rf_model.predict_proba(df)
        prob_gb = gb_model.predict_proba(df)

        clases_rf = rf_model.named_steps['classifier'].classes_
        clases_gb = gb_model.named_steps['classifier'].classes_

        barras_rf = generar_grafico_barras(prob_rf, "Random Forest - Probabilidades", clases_rf)
        barras_gb = generar_grafico_barras(prob_gb, "Gradient Boosting - Probabilidades", clases_gb)

        # Para simulación: usar predicciones como "y_true" en matriz
        matriz_rf = generar_matriz_confusion(pred_rf, pred_rf, "Matriz RF")
        matriz_gb = generar_matriz_confusion(pred_gb, pred_gb, "Matriz GB")

        return jsonify({
            "prediccion_random_forest": pred_rf,
            "prediccion_gradient_boosting": pred_gb,
            "grafico_barras_rf": barras_rf,
            "grafico_barras_gb": barras_gb,
            "matriz_confusion_rf": matriz_rf,
            "matriz_confusion_gb": matriz_gb
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
