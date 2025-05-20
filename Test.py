import numpy as np
import json
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === 1. Charger le RPC d'origine ===
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)

rpc_orig = RPCModel(data["rpc"], dict_format="rpcm")

# === 2. Position de la caméra déplacée (vue oblique) ===
inclination_distance_deg = 0.0004  # ~40 m de décalage lat/lon

# Exemple : déplacement vers le nord-est
cam_lon = rpc_orig.lon_offset + inclination_distance_deg
cam_lat = rpc_orig.lat_offset + inclination_distance_deg
cam_alt = rpc_orig.alt_offset + 50  # un peu plus haut

# Créer un faux RPC déplacé (même polynômes pour l’instant)
rpc_fake = RPCModel(data["rpc"], dict_format="rpcm")
rpc_fake.lon_offset = cam_lon
rpc_fake.lat_offset = cam_lat
rpc_fake.alt_offset = cam_alt

# === 3. Générer des points 3D en face de la caméra déplacée ===
# On centre la grille sur la nouvelle position
num_xy, num_z = 10, 3
delta_lon = rpc_orig.lon_scale * 0.2
delta_lat = rpc_orig.lat_scale * 0.2
delta_alt = rpc_orig.alt_scale * 0.05

lon_vals = np.linspace(-delta_lon, delta_lon, num_xy) + cam_lon
lat_vals = np.linspace(-delta_lat, delta_lat, num_xy) + cam_lat
alt_vals = np.linspace(-delta_alt, delta_alt, num_z) + rpc_orig.alt_offset

lon, lat, alt = np.meshgrid(lon_vals, lat_vals, alt_vals)
scene_points_3d = np.stack([lon.ravel(), lat.ravel(), alt.ravel()], axis=1)

# === 4. Projeter ces points avec le modèle déplacé (vue oblique simulée) ===
points_2d = np.array([rpc_fake.projection(*p) for p in scene_points_3d])
points_2d = np.flip(points_2d, axis=1)  # (col, row) → (row, col)

# === 5. Recalibrer un nouveau modèle RPC (vue oblique réelle) ===
rpc_new = calibrate_rpc(
    target=points_2d,
    input_locs=scene_points_3d,
    separate=True,
    orientation="projection",
    init=rpc_fake
)

# === 6. Convertir en JSON format RPCM ===
rpc_dict = {
    "row_offset": rpc_new.row_offset,
    "col_offset": rpc_new.col_offset,
    "lat_offset": rpc_new.lat_offset,
    "lon_offset": rpc_new.lon_offset,
    "alt_offset": rpc_new.alt_offset,
    "row_scale": rpc_new.row_scale,
    "col_scale": rpc_new.col_scale,
    "lat_scale": rpc_new.lat_scale,
    "lon_scale": rpc_new.lon_scale,
    "alt_scale": rpc_new.alt_scale,
    "row_num": list(rpc_new.row_num),
    "row_den": list(rpc_new.row_den),
    "col_num": list(rpc_new.col_num),
    "col_den": list(rpc_new.col_den)
}

# === 7. Sauvegarder le fichier RPC modifié ===
data["rpc"] = rpc_dict
output_path = "JAX_214_007_RPC_oblique_NE.json"
with open(output_path, "w") as f:
    json.dump(data, f, indent=2)

print(f"✅ Vue oblique générée et sauvegardée dans {output_path}")
