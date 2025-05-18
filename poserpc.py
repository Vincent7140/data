import json
import numpy as np
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel

# === Charger le fichier RPC original ===
with open("JAX_214_007_RGB.json") as f:
    original_data = json.load(f)

# === Charger le modèle RPC original ===
rpc = RPCModel(original_data["rpc"], dict_format="rpcm")

# === Générer des points 3D autour du centre ===
N = 500
center_lon, center_lat, center_alt = rpc.lon_offset, rpc.lat_offset, rpc.alt_offset
lon = center_lon + np.random.uniform(-rpc.lon_scale, rpc.lon_scale, N)
lat = center_lat + np.random.uniform(-rpc.lat_scale, rpc.lat_scale, N)
alt = center_alt + np.random.uniform(-rpc.alt_scale / 10, rpc.alt_scale / 10, N)
points_3d = np.stack([lon, lat, alt], axis=1)

# === Projeter les points avec le modèle original ===
points_2d = np.array([rpc.projection(*p) for p in points_3d])
points_2d = np.flip(points_2d, axis=1)  # (col, row) → (row, col)

# === Calibrer un nouveau modèle RPC ===
rpc_new = calibrate_rpc(
    target=points_2d,
    input_locs=points_3d,
    separate=True,
    orientation="projection",
    init=rpc
)

# === Convertir le modèle RPC calibré en dictionnaire ===
rpc_new_dict = {
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

# === Remplacer les anciens coefficients RPC dans le JSON original ===
original_data["rpc"] = rpc_new_dict

# === Sauvegarder un nouveau fichier JSON ===
with open("JAX_214_007_RGB_rpc_modifie.json", "w") as f:
    json.dump(original_data, f, indent=2)

print("✅ JSON mis à jour avec les nouveaux coefficients RPC.")
