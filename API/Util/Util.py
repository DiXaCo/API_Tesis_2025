import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

def plot_lime_custom(exp, num_features=10):
    features = exp.as_list(label=exp.available_labels()[0])[:num_features]
    nombres = [f[0] for f in features]
    impactos = [f[1] for f in features]

    colores = ['green' if val > 0 else 'red' for val in impactos]

    plt.figure(figsize=(5,6))
    plt.barh(nombres[::-1], impactos[::-1], color=colores[::-1])
    plt.title("Explicación LIME - Impacto características")
    plt.xlabel("Impacto en la predicción")
    plt.tight_layout()
    plt.show()


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

def plot_probabilidades_clases(probas, clases):

    plt.figure(figsize=(8,5))
    colores = ['blue' if c != 'predicho' else 'green' for c in clases] 

    plt.barh(clases, probas, color='skyblue')
    plt.xlabel("Probabilidad")
    plt.title("Probabilidades por clase")
    plt.xlim(0, 1)

    for i, v in enumerate(probas):
        plt.text(v + 0.01, i, f"{v:.2f}", color='black', va='center')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    return img_base64