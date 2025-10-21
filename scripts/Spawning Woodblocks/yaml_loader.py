import os
import yaml

def load_plates_from_yaml(path):
    """Lädt Platten aus YAML, konvertiert Breite, Tiefe, Dicke in Meter-Floats abhängig von der Einheit,
    bestimmt Orientierung und berechnet automatisch die Positionen nebeneinander."""
    
    with open(path, "r") as file:
        data = yaml.safe_load(file)

    # Einheit bestimmen (default: Meter)
    unit = data.get("unit", "m").lower()
    if unit == "mm":
        factor = 0.001
    elif unit == "cm":
        factor = 0.01
    elif unit == "m":
        factor = 1.0
    else:
        raise ValueError(f"Unbekannte Einheit: {unit}. Erlaubt: mm, cm, m.")

    # Orientation bestimmen
    orientation_type = data.get("orientation", "horizontal").lower()
    if orientation_type == "horizontal":
        ori = (0.0, 0.0, 0.0, 1.0)
    elif orientation_type == "vertical":
        ori = (0.0, 0.7071, 0.0, 0.7071)
    else:
        raise ValueError(f"Unbekannte Orientierung: {orientation_type}. Erlaubt: horizontal, vertical.")

    """plates = data.get("plates", [])
    if not plates:
        print("⚠️ Keine Platten im YAML gefunden!")
        return [], ori"""

    plates = data.get("plates", [])
    converted_plates = []

    for plate in plates:
        # Konvertierung in Meter
        width = float(plate["width"]) * factor
        depth = float(plate["depth"]) * factor
        thickness = float(plate["thickness"]) * factor
        pos_yaml = plate.get("position", [0.0, 0.0, 0.0])

        # Offset für vertikal vs. horizontal
        if orientation_type == "horizontal":
            z_offset = thickness / 2
        else:
            z_offset = width / 2  # steht hochkant

        # Sicherheitsabstand
        z_offset += 0.001

        # Position konvertieren + Offset
        pos = (
            pos_yaml[0]*factor,
            pos_yaml[1]*factor,
            pos_yaml[2]*factor + z_offset,
                )
        
        # Neue Plate-Daten erstellen
        new_plate = dict(plate)
        new_plate.update({
            "width": width,
            "depth": depth,
            "thickness": thickness,
            "position": pos
        })
        converted_plates.append(new_plate)

    return converted_plates, ori


# automatisches Laden beim Import
script_dir = os.path.dirname(__file__)
yaml_path = os.path.abspath(os.path.join(script_dir, "../..", "config", "plates.yaml"))
plates, orientation_plates = load_plates_from_yaml(yaml_path)
