
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lime.lime_tabular
import numpy as np
import pandas as pd

def generar_explica_lime(modelo, datos_df, used_features):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(datos_df),
        feature_names=used_features,
        class_names=modelo.classes_,
        mode="classification",
        discretize_continuous=True
    )

    exp = explainer.explain_instance(
        datos_df.iloc[0],
        lambda x: modelo.predict_proba(pd.DataFrame(x, columns=used_features)),
        num_features=10
    )

    fig = exp.as_pyplot_figure()
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_b64

def construir_tabla_lime(explicacion, datos_df):
    lista = []
    for feature, peso in explicacion.as_list():
        nombre = feature.split()[0].strip()
        valor_np = datos_df[nombre].values[0] if nombre in datos_df else "?"
        valor = valor_np.item() if hasattr(valor_np, "item") else valor_np
        influencia = "Positiva" if peso >= 0 else "Negativa"
        lista.append({
            "caracteristica": feature,
            "valor": valor,
            "influencia": influencia
        })
    return lista
