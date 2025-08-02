# Util/reglas_lime.py

def convert_to_if_then(explanation, class_name, instance_to_explain: str):
    rule_parts = []

    try:
        for feature_name, importance in explanation.as_list(label=class_name):
            if importance > 0:
                rule_part = f"{feature_name} tiene una importancia de {importance:.4f}"
                rule_parts.append(rule_part)

        if rule_parts:
            combined_rule = (
                f"Con las condiciones actuales, se recomienda hacer una contratación en {instance_to_explain}. "
                "Las siguientes características influyen: " + ", ".join(rule_parts) + "."
            )
        else:
            combined_rule = (
                f"Con las condiciones actuales, se recomienda hacer una contratación en {instance_to_explain}. "
                "No se encontraron características con importancia positiva."
            )

    except KeyError:
        combined_rule = (
            f"Con las condiciones actuales, se recomienda hacer una contratación en {instance_to_explain}. "
            "No se encontraron explicaciones para esta clase."
        )

    return [combined_rule]