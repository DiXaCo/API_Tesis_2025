import io
import base64
import matplotlib.pyplot as plt
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

def generar_explica_lime(modelo, df, used_features):
    """
    Genera una imagen LIME estilo clásico con barras horizontales por clase (como LIME_RF.png y LIME_GB.png).
    Retorna una imagen PNG en base64 para incrustar en HTML.
    """
    # Crear el explicador
    explainer = LimeTabularExplainer(
        training_data=df.values,
        feature_names=used_features,
        class_names=modelo.classes_,
        mode='classification'
    )

    # Función de predicción para LIME
    def predict_fn(x_array):
        df_temp = pd.DataFrame(x_array, columns=used_features)
        return modelo.predict_proba(df_temp)

    # Generar explicación
    exp = explainer.explain_instance(
        df.iloc[0].values,
        predict_fn,
        num_features=len(used_features)
    )

    # 🔥 Aquí generamos el gráfico de LIME tipo barras horizontales por clase
    fig = exp.as_pyplot_figure()

    # Convertir a imagen base64
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    lime_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return lime_base64


def construir_tabla_lime(exp, df):
    """
    Construye una tabla resumen de las características más influyentes 
    según LIME para una predicción dada.

    Args:
        exp: Objeto Explanation de LIME.
        df: DataFrame con la instancia evaluada (una sola fila).

    Returns:
        Lista de diccionarios con las columnas: caracteristica, valor, influencia.
    """
    explicaciones = exp.as_list()
    fila = df.iloc[0]
    tabla = []

    for feature_str, peso in explicaciones:
        # Intentar extraer el nombre de la característica
        # Ej: 'CAE_TYPE_5 > 0.5' → 'CAE_TYPE_5'
        tokens = feature_str.split()
        nombre = tokens[0]

        # Si hay paréntesis u otros símbolos, eliminarlos
        nombre = nombre.strip("()")

        # Obtener valor real de la característica
        valor = fila.get(nombre, "N/A")

        # Determinar tipo de influencia
        influencia = "Positiva" if peso >= 0 else "Negativa"

        tabla.append({
            "caracteristica": nombre,
            "valor": valor,
            "influencia": influencia
        })

    return tabla
