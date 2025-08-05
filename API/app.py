from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import os
import warnings
import json
import pandas as pd
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

from Util.Util import generar_explica_lime, construir_tabla_lime
from Util.reglas_lime import convert_to_if_then

warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
CORS(app, origins=["http://localhost:8000"], supports_credentials=True)

app.secret_key = 'clave_super_segura_para_session'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "model", "modelos_experimento_B.pkl")
USUARIOS_PATH = os.path.join(BASE_DIR, "users.json")

# Cargar modelos
modelos = joblib.load(MODELO_PATH)
used_features = modelos['random_forest'].used_features
clases = modelos['random_forest'].classes_

# Funciones de gestión de usuarios
def cargar_usuarios():
    if os.path.exists(USUARIOS_PATH):
        with open(USUARIOS_PATH, "r") as f:
            return json.load(f)
    return {}

def guardar_usuarios(usuarios):
    with open(USUARIOS_PATH, "w") as f:
        json.dump(usuarios, f, indent=2)

# Rutas
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])
        usuarios = cargar_usuarios()
        if email in usuarios:
            return "Usuario ya existe", 400
        usuarios[email] = {"password": password}
        guardar_usuarios(usuarios)
        return redirect("/login?mensaje=Registro%20exitoso,%20puedes%20iniciar%20sesión")
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    mensaje = request.args.get("mensaje")  # <- Captura mensajes opcionales
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        usuarios = cargar_usuarios()
        if email in usuarios and check_password_hash(usuarios[email]["password"], password):
            session["usuario"] = email
            return redirect("/")
        return "Credenciales incorrectas", 401
    return render_template("login.html", mensaje=mensaje)


@app.route("/logout")
def logout():
    session.pop("usuario", None)
    return redirect("/login?mensaje=Sesión%20cerrada%20correctamente")


@app.route("/", methods=["GET"])
def index():
    if "usuario" not in session:
        return redirect("/login")
    return render_template("formulario.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        modelo_nombre = data.get("modelo")
        entrada = data.get("entrada", {})

        entrada["MAIN_ACTIVITY_general public\services"] = entrada.pop("MAIN_ACTIVITY_general_public_services", 0)

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

        from lime.lime_tabular import LimeTabularExplainer
        def predict_proba_wrapper(x_array):
            df_temp = pd.DataFrame(x_array, columns=used_features)
            return modelo.predict_proba(df_temp)

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

        tabla_lime = construir_tabla_lime(exp, df)

        # Convertir valores a tipos nativos de Python
        for fila in tabla_lime:
            if hasattr(fila["valor"], "item"):
                fila["valor"] = fila["valor"].item()

        return jsonify({
            "prediccion": str(pred[0]),
            "probabilidad": round(float(proba), 4),
            "explicacion_lime": lime_img,
            "reglas_por_clase": reglas_texto,
            "tabla_lime": tabla_lime
        })












    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno", "mensaje": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
