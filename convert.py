import pandas as pd
import numpy as np
import os

def convertir_angles_en_degres(fichier_entree, colonnes_angles=None, sep_auto=True):
    # Déterminer le séparateur : auto (espace ou virgule)
    if sep_auto:
        with open(fichier_entree, 'r') as f:
            premiere_ligne = f.readline()
        sep = ',' if ',' in premiere_ligne else r'\s+'
    else:
        sep = r'\s+'

    # Lire le fichier
    df = pd.read_csv(fichier_entree, sep=sep, engine='python')

    # Colonnes à convertir (par défaut)
    if colonnes_angles is None:
        colonnes_angles = ['Az', 'El', 'Azs', 'Els']

    # Vérifier les colonnes présentes
    colonnes_valides = [col for col in colonnes_angles if col in df.columns]

    # Convertir les radians en degrés
    df[colonnes_valides] = np.degrees(df[colonnes_valides])
    df[colonnes_valides] = df[colonnes_valides].round(4)

    # Nom du fichier de sortie
    nom_fichier = os.path.splitext(os.path.basename(fichier_entree))[0]
    fichier_sortie = f"output_{nom_fichier}.txt"

    # Sauvegarder
    df.to_string(open(fichier_sortie, 'w'), index=False)

    print(f"✅ Conversion terminée. Fichier sauvegardé : {fichier_sortie}")

# Exemple d'utilisation :
convertir_angles_en_degres("JAX_260_df1_md.txt")
