from flask import Flask, request, jsonify, render_template, session, redirect
from flask_cors import CORS
import os
import warnings
import json
import pandas as pd
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
import matplotlib.pyplot as plt
import io
import base64
import re
from Util.Util import plot_lime_custom, construir_tabla_lime, plot_probabilidades_clases
from Util.reglas_lime import convert_to_if_then
#-----------------------------------------------------------------------------------------------------


def generar_sugerencia_group_duration(prediccion_intervalo, probabilidades=None):
    """Convierte intervalo de predicción en sugerencias prácticas de duración"""
    import re
    
    # DEBUG: Ver qué estamos recibiendo
    print(f"🔍 DEBUG: prediccion_intervalo = '{prediccion_intervalo}'")
    print(f"🔍 DEBUG: type(prediccion_intervalo) = {type(prediccion_intervalo)}")
    print(f"🔍 DEBUG: probabilidades = {probabilidades}")
    
    try:
        # Extraer números del intervalo
        numeros = re.findall(r"[-+]?\d*\.?\d+", prediccion_intervalo)
        print(f"🔍 DEBUG: números encontrados = {numeros}")
        
        if len(numeros) >= 2:
            limite_inf = max(0, float(numeros[0]))  # No duraciones negativas
            limite_sup = float(numeros[1])
            
            print(f"🔍 DEBUG: limite_inf = {limite_inf}, limite_sup = {limite_sup}")
            
            # Calcular opciones
            minimo = round(limite_inf, 0)
            promedio = round((limite_inf + limite_sup) / 2, 0)
            maximo = round(limite_sup, 0)
            
            print(f"🔍 DEBUG: minimo = {minimo}, promedio = {promedio}, maximo = {maximo}")
            
            # Generar sugerencias múltiples
            sugerencias = {
                "opcion_recomendada": {
                    "valor": promedio,
                    "unidad": "meses",
                    "descripcion": f"Duración equilibrada ({promedio} meses)",
                    "justificacion": "Basado en el punto medio del rango predicho"
                },
                "alternativas": [
                    {
                        "tipo": "Conservadora",
                        "valor": minimo,
                        "unidad": "meses", 
                        "descripcion": f"Duración mínima ({minimo} meses)",
                        "uso_recomendado": "Para proyectos con alcance bien definido"
                    },
                    {
                        "tipo": "Estándar",
                        "valor": promedio,
                        "unidad": "meses",
                        "descripcion": f"Duración promedio ({promedio} meses)", 
                        "uso_recomendado": "Para la mayoría de casos"
                    },
                    {
                        "tipo": "Extendida",
                        "valor": maximo,
                        "unidad": "meses",
                        "descripcion": f"Duración máxima ({maximo} meses)",
                        "uso_recomendado": "Para proyectos complejos con incertidumbre"
                    }
                ],
                "rango_original": {
                    "minimo": limite_inf,
                    "maximo": limite_sup,
                    "unidad": "meses"
                }
            }
            
            # Añadir equivalencias en diferentes unidades
            for alt in sugerencias["alternativas"]:
                alt["equivalencias"] = {
                    "dias": round(alt["valor"] * 30.44),
                    "años": round(alt["valor"] / 12, 1)
                }
            
            # Añadir recomendaciones según duración
            recomendaciones = []
            if promedio <= 6:
                recomendaciones = [
                    "📋 Contrato de corta duración - definir entregables específicos",
                    "💰 Considerar pagos por hitos",
                    "🔄 Incluir opción de renovación si es necesario"
                ]
            elif promedio <= 24:
                recomendaciones = [
                    "📊 Incluir evaluaciones trimestrales",
                    "💱 Considerar cláusulas de ajuste de precios",
                    "📋 Definir procedimientos de modificación"
                ]
            else:
                recomendaciones = [
                    "🎯 Establecer hitos semestrales de evaluación",
                    "💼 Incluir garantías de cumplimiento",
                    "📈 Cláusulas de revisión anual"
                ]
                
            sugerencias["recomendaciones_contractuales"] = recomendaciones
            
            # Añadir nivel de confianza
            if probabilidades:
                max_prob = max(probabilidades)
                sugerencias["confianza"] = {
                    "nivel": "Alta" if max_prob > 0.7 else "Media" if max_prob > 0.4 else "Baja",
                    "porcentaje": f"{max_prob:.1%}",
                    "interpretacion": "Predicción confiable" if max_prob > 0.6 else "Considerar análisis adicional"
                }
            
            print(f"🔍 DEBUG: Retornando sugerencias calculadas correctamente")
            return sugerencias
            
        else:
            # Fallback si no se puede parsear el intervalo
            print(f"⚠️ DEBUG: No se encontraron suficientes números. len(numeros) = {len(numeros)}")
            return {
                "opcion_recomendada": {
                    "valor": 12,
                    "unidad": "meses",
                    "descripcion": "Duración estándar por defecto",
                    "justificacion": "No se pudo determinar el rango específico"
                }
            }
            
    except Exception as e:
        print(f"❌ ERROR generando sugerencias: {e}")
        import traceback
        traceback.print_exc()
        return {
            "opcion_recomendada": {
                "valor": 12,
                "unidad": "meses", 
                "descripcion": "Duración estándar por defecto",
                "justificacion": "Error en el procesamiento de la predicción"
            }
        }



#-------------------------------------------------------------------------------------------
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


# Traducción de clase a duración en meses (según ANNEX II)
DURACION_ESTIMADA_POR_CLASE = {
    "short": 3.0,    # ajusta según las clases reales de tu modelo
    "medium": 12.0,
    "long": 36.0
}


# Mapeo completo de nombres frontend a backend
CAMPO_MAPEO_COMPLETO = {
    "MULTIPLE_CONTRACTING": "B_MULTIPLE_CAE_n",
    "ACTING_ON_BEHALF": "B_ON_BEHALF_n",
    "WORKS_CONTRACT": "TYPE_OF_CONTRACT_w",
    "ISO_COUNTRY_CODE_SI": "ISO_COUNTRY_CODE_si",
    "ISO_COUNTRY_CODE_LU": "ISO_COUNTRY_CODE_lu",
    "NUMBER_OF_CONTRACTS": "NUMBER_AWARDS",
    "NUMBER_OF_LOTS": "LOTS_NUMBER",
    "NUMBER_OF_OFFERS": "NUMBER_OFFERS",
    "NUMBER_OFFERS_SME": "NUMBER_TENDERS_SME",
    "MAIN_ACTIVITY_health": "MAIN_ACTIVITY_health",
    "MAIN_ACTIVITY_general_public_services": "MAIN_ACTIVITY_general_public_services",
    "CAE_TYPE_3": "CAE_TYPE_3",
    "CAE_TYPE_4": "CAE_TYPE_4",
    "CAE_TYPE_5": "CAE_TYPE_5",
    "GROUP_CPV_15": "GROUP_CPV_15",
    "GROUP_CPV_33": "GROUP_CPV_33",
    "GROUP_CPV_45": "GROUP_CPV_45",
}



# Funciones de gestión de usuarios
def cargar_usuarios():
    if os.path.exists(USUARIOS_PATH):
        with open(USUARIOS_PATH, "r") as f:
            return json.load(f)
    return {}

def guardar_usuarios(usuarios):
    with open(USUARIOS_PATH, "w") as f:
        json.dump(usuarios, f, indent=2)



# --------------------------------------------
# FUNCION para traducir etiquetas a formato legible
def interpretar_etiqueta_duracion(etiqueta):
    """
    Convierte una etiqueta como '(-0.2, 67.611]' en 'Duración estimada entre 0.2 y 67.6 meses'
    """
    try:
        if etiqueta.startswith("(") or etiqueta.startswith("["):
            etiqueta = etiqueta.replace("(", "").replace("]", "")
            partes = etiqueta.split(",")
            if len(partes) == 2:
                inicio = round(float(partes[0]), 1)
                fin = round(float(partes[1]), 1)
                return f"Duración estimada entre {inicio} y {fin} meses"
        elif etiqueta.replace(".", "").isnumeric():
            return f"Duración estimada: {round(float(etiqueta), 1)} meses"
        else:
            return f"Duración estimada: {etiqueta}"
    except:
        return "Duración estimada desconocida"


# Rutas para registro/login/logout/index
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
    mensaje = request.args.get("mensaje")
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
        entrada_raw = data.get("entrada", {})

        print("📥 Entrada cruda desde el frontend:")
        print(json.dumps(entrada_raw, indent=2))

        # Mapear nombres frontend -> backend
        entrada_mapeada = {}
        for clave_frontend, valor in entrada_raw.items():
            clave_backend = CAMPO_MAPEO_COMPLETO.get(clave_frontend, clave_frontend)
            entrada_mapeada[clave_backend] = valor

        print("\n🔁 Entrada mapeada a nombres del modelo:")
        print(json.dumps(entrada_mapeada, indent=2))

        # Completar con 0 campos faltantes
        for campo in used_features:
            if campo not in entrada_mapeada:
                print(f"⚠️ Campo faltante detectado: {campo}, se establece en 0")
                entrada_mapeada[campo] = 0

        # Mostrar todos los campos finales
        print("\n✅ Entrada final completa (lista para el modelo):")
        print(json.dumps(entrada_mapeada, indent=2))

        # Validar campos requeridos
        faltantes = [f for f in used_features if f not in entrada_mapeada]
        if faltantes:
            print(f"❌ Faltan campos requeridos: {faltantes}")
            return jsonify({
                "error": "Faltan campos requeridos para la predicción.",
                "campos_faltantes": faltantes
            }), 400

        # Crear dataframe para modelo
        df = pd.DataFrame([entrada_mapeada])[used_features]

        modelo = modelos.get(modelo_nombre)
        if modelo is None:
            return jsonify({"error": "Modelo no reconocido"}), 400

        # Predicción
        pred = modelo.predict(df)
        probas = modelo.predict_proba(df)[0]

        indice_pred = list(modelo.classes_).index(pred[0])
        proba = probas[indice_pred]


        # Interpretar etiqueta como duración legible
        etiqueta = str(pred[0])
        mensaje_duracion = interpretar_etiqueta_duracion(etiqueta)



        print(f"\n🔮 Predicción: {pred[0]}, Probabilidad: {proba:.4f}")

        # LIME
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

        img_probas_base64 = plot_probabilidades_clases(probas, modelo.classes_)

        reglas_texto = {}
        for clase in exp.available_labels():
            reglas_texto[str(clase)] = convert_to_if_then(exp, clase, pred[0])

        tabla_lime = construir_tabla_lime(exp, df)
        for fila in tabla_lime:
            if hasattr(fila["valor"], "item"):
                fila["valor"] = fila["valor"].item()

        print("\n📊 Tabla LIME generada:")
        for fila in tabla_lime:
            print(fila)

     #   return jsonify({
      #      "prediccion": str(pred[0]),
       #     "probabilidad": round(float(proba), 4),
        #    "grafico_probabilidades_base64": img_probas_base64,
         #   "reglas_por_clase": reglas_texto,
          #  "tabla_lime": tabla_lime
        #})
    

       

        etiqueta = str(pred[0])
        try:
            # Buscar números dentro del intervalo, como en (-0.2, 67.611]
            numeros = re.findall(r"[-+]?\d*\.\d+|\d+", etiqueta)
            if len(numeros) == 2:
                inicio = round(float(numeros[0]), 1)
                fin = round(float(numeros[1]), 1)
                mensaje_duracion = f"Duración estimada entre {inicio} y {fin} meses"
            else:
                # Si no es un intervalo, tratar de convertir directamente
                duracion = round(float(etiqueta), 1)
                mensaje_duracion = f"Duración estimada de {duracion} meses"
        except:
            mensaje_duracion = "Duración estimada desconocida"

        mensaje_explicativo = f"Según el análisis de las características ingresadas, se estima que la duración del contrato será de {mensaje_duracion}. Esta recomendación considera factores como el tipo de contrato, número de ofertas, y la actividad principal de la autoridad contratante."

       # Mensaje explicativo con rango
        mensaje_explicativo = (
            f"Según el análisis de las características ingresadas, se estima que la duración del contrato será de "
            f"{mensaje_duracion}. Esta recomendación considera factores como el tipo de contrato, número de ofertas, "
            f"y la actividad principal de la autoridad contratante."
        )
#-----------------------------------------------------------------------
        # Justo antes de llamar generar_sugerencia_group_duration
        print(f"🔍 DEBUG MAIN: pred[0] = '{pred[0]}', type = {type(pred[0])}")
        sugerencias_duracion = generar_sugerencia_group_duration(str(pred[0]), probas)
                    
        sugerencias_duracion = generar_sugerencia_group_duration(str(pred[0]), probas)
#-------------------------------------------------------------------------


        return jsonify({
            "prediccion": str(pred[0]),
            "probabilidad": round(float(proba), 4),
            "grafico_probabilidades_base64": img_probas_base64,
            "reglas_por_clase": reglas_texto,
            "tabla_lime": tabla_lime,
            "duracion_meses_aproximada": mensaje_duracion,
            "mensaje_explicativo": mensaje_explicativo,

            #------------------------------------
            "sugerencias_duracion": sugerencias_duracion,
            "sugerencia_principal": {
            "valor_meses": sugerencias_duracion["opcion_recomendada"]["valor"],
            "descripcion": sugerencias_duracion["opcion_recomendada"]["descripcion"]}

         })


    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno", "mensaje": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
