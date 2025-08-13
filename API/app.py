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

def validar_entrada_modelo(entrada_mapeada):
    """
    Validar que al menos una categoría esté seleccionada por grupo
    Aplica valores por defecto inteligentes para evitar configuraciones imposibles
    """
    print("🔧 Validando entrada del modelo...")
    
    # 1. Verificar actividad principal (al menos una debe estar seleccionada)
    actividades = ['MAIN_ACTIVITY_health', 'MAIN_ACTIVITY_general_public_services']
    actividades_seleccionadas = sum(entrada_mapeada.get(act, 0) for act in actividades)
    
    if actividades_seleccionadas == 0:
        print("⚠️ No hay actividad principal seleccionada. Aplicando por defecto: Servicios públicos generales")
        entrada_mapeada['MAIN_ACTIVITY_general_public_services'] = 1
    elif actividades_seleccionadas > 1:
        print("⚠️ Múltiples actividades principales seleccionadas. Manteniendo solo la primera.")
        # Mantener solo la primera encontrada
        primera_encontrada = False
        for act in actividades:
            if entrada_mapeada.get(act, 0) == 1 and not primera_encontrada:
                primera_encontrada = True
            elif entrada_mapeada.get(act, 0) == 1:
                entrada_mapeada[act] = 0
    
    # 2. Verificar tipo de autoridad (CAE_TYPE)
    tipos_cae = ['CAE_TYPE_3', 'CAE_TYPE_4', 'CAE_TYPE_5']
    tipos_seleccionados = sum(entrada_mapeada.get(tipo, 0) for tipo in tipos_cae)
    
    if tipos_seleccionados == 0:
        print("⚠️ No hay tipo de autoridad seleccionado. Aplicando por defecto: CAE_TYPE_4")
        entrada_mapeada['CAE_TYPE_4'] = 1
    elif tipos_seleccionados > 1:
        print("⚠️ Múltiples tipos de autoridad seleccionados. Manteniendo solo el primero.")
        primera_encontrada = False
        for tipo in tipos_cae:
            if entrada_mapeada.get(tipo, 0) == 1 and not primera_encontrada:
                primera_encontrada = True
            elif entrada_mapeada.get(tipo, 0) == 1:
                entrada_mapeada[tipo] = 0
    
    # 3. Verificar grupo CPV (clasificación de actividad)
    grupos_cpv = ['GROUP_CPV_15', 'GROUP_CPV_33', 'GROUP_CPV_45']
    grupos_seleccionados = sum(entrada_mapeada.get(grupo, 0) for grupo in grupos_cpv)
    
    if grupos_seleccionados == 0:
        print("⚠️ No hay clasificación de actividad seleccionada. Aplicando por defecto: GROUP_CPV_15")
        entrada_mapeada['GROUP_CPV_15'] = 1
    # Nota: Para CPV, pueden haber múltiples seleccionados (es más realista)
    
    # 4. Verificar país (ISO_COUNTRY_CODE)
    paises = ['ISO_COUNTRY_CODE_lu', 'ISO_COUNTRY_CODE_si']
    paises_seleccionados = sum(entrada_mapeada.get(pais, 0) for pais in paises)
    
    if paises_seleccionados == 0:
        print("⚠️ No hay país seleccionado. Aplicando por defecto: Luxemburgo")
        entrada_mapeada['ISO_COUNTRY_CODE_lu'] = 1
    elif paises_seleccionados > 1:
        print("⚠️ Múltiples países seleccionados. Manteniendo solo el primero.")
        primera_encontrada = False
        for pais in paises:
            if entrada_mapeada.get(pais, 0) == 1 and not primera_encontrada:
                primera_encontrada = True
            elif entrada_mapeada.get(pais, 0) == 1:
                entrada_mapeada[pais] = 0
    
    # 5. Validar valores numéricos
    campos_numericos = ['NUMBER_AWARDS', 'LOTS_NUMBER', 'NUMBER_OFFERS', 'NUMBER_TENDERS_SME']
    for campo in campos_numericos:
        valor = entrada_mapeada.get(campo, 0)
        if valor < 0:
            print(f"⚠️ Valor negativo en {campo}. Estableciendo en 0.")
            entrada_mapeada[campo] = 0
        elif valor == 0 and campo in ['NUMBER_AWARDS', 'LOTS_NUMBER', 'NUMBER_OFFERS']:
            print(f"⚠️ Valor 0 en {campo}. Estableciendo en 1 (mínimo realista).")
            entrada_mapeada[campo] = 1
    
    # 6. Lógica de negocio: NUMBER_TENDERS_SME no puede ser mayor que NUMBER_OFFERS
    if entrada_mapeada.get('NUMBER_TENDERS_SME', 0) > entrada_mapeada.get('NUMBER_OFFERS', 0):
        print("⚠️ Número de ofertas PYMEs mayor que total de ofertas. Corrigiendo.")
        entrada_mapeada['NUMBER_TENDERS_SME'] = entrada_mapeada.get('NUMBER_OFFERS', 0)
    
    print("✅ Validación completada")
    return entrada_mapeada


def generar_sugerencia_group_duration(prediccion_intervalo, probabilidades=None):
    """Convierte intervalo de predicción en sugerencias prácticas de duración"""
    import re
    import numpy as np
    
    try:
        # Extraer números del intervalo
        numeros = re.findall(r"[-+]?\d*\.?\d+", prediccion_intervalo)
        
        if len(numeros) >= 2:
            limite_inf = max(0, float(numeros[0]))  # No duraciones negativas
            limite_sup = float(numeros[1])
            
            # Calcular opciones
            minimo = round(limite_inf, 0)
            promedio = round((limite_inf + limite_sup) / 2, 0)
            maximo = round(limite_sup, 0)
            
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
            
            # Añadir equivalencias en diferentes unidades (usando 30.4 días/mes como en tu metodología)
            for alt in sugerencias["alternativas"]:
                alt["equivalencias"] = {
                    "dias": round(alt["valor"] * 30.4),  # Usar 30.4 como en tu cálculo de duración
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
            
            # Verificar probabilidades de forma segura para arrays NumPy
            if probabilidades is not None and hasattr(probabilidades, '__len__') and len(probabilidades) > 0:
                try:
                    # Convertir a lista Python si es array NumPy para evitar ambigüedad
                    if hasattr(probabilidades, 'tolist'):
                        probs_list = probabilidades.tolist()
                    else:
                        probs_list = list(probabilidades)
                    
                    # Ahora sí podemos usar max() sin problemas
                    max_prob = max(probs_list)
                    
                    sugerencias["confianza"] = {
                        "nivel": "Alta" if max_prob > 0.7 else "Media" if max_prob > 0.4 else "Baja",
                        "porcentaje": f"{max_prob:.1%}",
                        "interpretacion": "Predicción confiable" if max_prob > 0.6 else "Considerar análisis adicional"
                    }
                    
                except Exception as prob_error:
                    # Fallback para confianza
                    sugerencias["confianza"] = {
                        "nivel": "Media",
                        "porcentaje": "N/A",
                        "interpretacion": "Error calculando confianza"
                    }
            else:
                sugerencias["confianza"] = {
                    "nivel": "Media",
                    "porcentaje": "N/A", 
                    "interpretacion": "Probabilidades no disponibles"
                }
            
            return sugerencias
            
        else:
            # Fallback si no se puede parsear el intervalo
            return {
                "opcion_recomendada": {
                    "valor": 12,
                    "unidad": "meses",
                    "descripcion": "Duración estándar por defecto",
                    "justificacion": "No se pudo determinar el rango específico"
                }
            }
            
    except Exception as e:
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

print("✅ Modelos cargados correctamente")
print(f"📊 Features disponibles: {len(used_features)}")
print(f"🎯 Clases del modelo: {len(clases)}")

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

        # 🔧 Remover llave vacía si llega desde el frontend
        if "" in entrada_raw:
            entrada_raw.pop("")

        print("📥 Entrada cruda desde el frontend:")
        print(json.dumps(entrada_raw, indent=2))

        # Mapear nombres frontend -> backend
        entrada_mapeada = {}
        for clave_frontend, valor in entrada_raw.items():
            clave_backend = CAMPO_MAPEO_COMPLETO.get(clave_frontend, clave_frontend)
            entrada_mapeada[clave_backend] = valor

        # ======= VALIDACIÓN DE ENTRADA NUEVA =======
        entrada_mapeada = validar_entrada_modelo(entrada_mapeada)
        # ==========================================

        # ======= SELECCIÓN DEL MODELO Y SUS FEATURES =======
        modelo = modelos.get(modelo_nombre)
        if modelo is None:
            return jsonify({"error": "Modelo no reconocido"}), 400

        used_features_modelo = getattr(modelo, "used_features", None)
        if not used_features_modelo:
            used_features_modelo = used_features
        # ====================================================

        # Completar con 0 campos faltantes
        for campo in used_features_modelo:
            if campo not in entrada_mapeada:
                entrada_mapeada[campo] = 0
        print("✅ Entrada final validada (lista para el modelo):")
        print(json.dumps(entrada_mapeada, indent=2))

        # Crear dataframe con el orden correcto para el modelo
        df = pd.DataFrame([entrada_mapeada])[used_features_modelo]
       
        # DEBUG
        print(f"\n🔍 DEBUG ENTRADA AL MODELO:")
        print(f"Modelo: {modelo_nombre}")
        print(f"DataFrame shape: {df.shape}")
        print(f"Valores de entrada:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")

        # DEBUG
        print(f"\n🔍 DEBUG CRÍTICO:")
        print(f"Archivo del modelo: {MODELO_PATH}")
        print(f"Modelo seleccionado: {modelo_nombre}")
        print(f"Tipo de modelo: {type(modelo)}")
        print(f"DataFrame final:")
        for col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")

        # Verificar que es el modelo balanceado
        if hasattr(modelo, 'class_weight'):
            print(f"Class weight: {modelo.class_weight}")
        else:
            print("⚠️ Modelo NO tiene class_weight - podría ser el archivo anterior")

        # Sugerencia
        pred = modelo.predict(df)
        probas = modelo.predict_proba(df)[0]
        indice_pred = list(modelo.classes_).index(pred[0])
        proba = probas[indice_pred]

        etiqueta = str(pred[0])
        mensaje_duracion = interpretar_etiqueta_duracion(etiqueta)

        print(f"🔮 Sugerencia: {pred[0]}, Probabilidad: {proba:.4f}")

        # =================== LIME ===================
        from lime.lime_tabular import LimeTabularExplainer

        def predict_proba_wrapper(x_array):
            df_temp = pd.DataFrame(x_array, columns=used_features_modelo)
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
        # ==============================================

        img_probas_base64 = plot_probabilidades_clases(probas, modelo.classes_)

        reglas_texto = {}
        for clase in exp.available_labels():
            reglas_texto[str(clase)] = convert_to_if_then(exp, clase, pred[0])

        tabla_lime = construir_tabla_lime(exp, df)
        for fila in tabla_lime:
            if hasattr(fila["valor"], "item"):
                fila["valor"] = fila["valor"].item()

        # Procesar etiqueta como rango o valor numérico
        etiqueta = str(pred[0])
        try:
            numeros = re.findall(r"[-+]?\d*\.\d+|\d+", etiqueta)
            if len(numeros) == 2:
                inicio = round(float(numeros[0]), 1)
                fin = round(float(numeros[1]), 1)
                mensaje_duracion = f"Duración estimada entre {inicio} y {fin} meses"
            else:
                duracion = round(float(etiqueta), 1)
                mensaje_duracion = f"Duración estimada de {duracion} meses"
        except:
            mensaje_duracion = "Duración estimada desconocida"

        mensaje_explicativo = (
            f"Según el análisis de las características ingresadas, se estima que la duración del contrato será de "
            f"{mensaje_duracion}. Esta recomendación considera factores como el tipo de contrato, número de ofertas, "
            f"y la actividad principal de la autoridad contratante."
        )

        sugerencias_duracion = generar_sugerencia_group_duration(str(pred[0]), probas)

        return jsonify({
            "prediccion": str(pred[0]),
            "probabilidad": round(float(proba), 4),
            "grafico_probabilidades_base64": img_probas_base64,
            "reglas_por_clase": reglas_texto,
            "tabla_lime": tabla_lime,
            "duracion_meses_aproximada": mensaje_duracion,
            "mensaje_explicativo": mensaje_explicativo,
            "sugerencias_duracion": sugerencias_duracion,
            "sugerencia_principal": {
                "valor_meses": sugerencias_duracion["opcion_recomendada"]["valor"],
                "descripcion": sugerencias_duracion["opcion_recomendada"]["descripcion"]
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": "Error interno", "mensaje": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)