import json
import numpy as np
import rasterio
from pyproj import Transformer
from rpcm.rpc_model import RPCModel

# === PARAMÈTRES ===
json_path = "JAX_214_007_RGB.json"
dsm_path = "JAX_214_007_RGB_dsm.tif"

# === 1. Charger le JSON et le modèle RPC ===
with open(json_path) as f:
    data = json.load(f)
rpc = RPCModel(data["rpc"], dict_format="rpcm")

# === 2. Extraire les points 2D (col, row) ===
points_2d = np.array(data["keypoints"]["2d_coordinates"])  # shape (N, 2)
cols = points_2d[:, 0].astype(int)
rows = points_2d[:, 1].astype(int)

# === 3. Charger le DSM et en extraire les altitudes ===
with rasterio.open(dsm_path) as dsm:
    alt_map = dsm.read(1)
    height, width = alt_map.shape

# Filtrer les points dans les limites de l’image
valid_mask = (cols >= 0) & (cols < width) & (rows >= 0) & (rows < height)
cols = cols[valid_mask]
rows = rows[valid_mask]
alts = alt_map[rows, cols]

# === 4. Localisation via RPC: (col, row, alt) -> (lon, lat) ===
lons, lats = rpc.localization(cols, rows, alts)

# === 5. Conversion en ECEF (coordonnées 3D réelles) ===
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
xs, ys, zs = transformer.transform(lons, lats, alts)
points_3d_ecef = np.vstack([xs, ys, zs]).T

print("Exemple de points 3D (ECEF):")
print(points_3d_ecef[:5])
