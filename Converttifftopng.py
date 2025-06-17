import os
from PIL import Image

# Répertoire source contenant les .tiff
source_dir = "chemin/vers/tes/tiff"

# Répertoire de destination pour les .png
dest_dir = "chemin/vers/tes/png"
os.makedirs(dest_dir, exist_ok=True)

# Parcourir tous les fichiers du dossier source
for filename in os.listdir(source_dir):
    if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
        tiff_path = os.path.join(source_dir, filename)
        png_filename = os.path.splitext(filename)[0] + ".png"
        png_path = os.path.join(dest_dir, png_filename)

        # Ouverture et conversion
        with Image.open(tiff_path) as img:
            img.convert("RGB").save(png_path, "PNG")

        print(f"Converti : {filename} -> {png_filename}")
