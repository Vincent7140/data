import json
import numpy as np
import cv2
import rasterio
from rpcm.rpc_model import RPCModel
from rpcfit.rpc_fit import calibrate_rpc
from rasterio.transform import xy
from pyproj import Transformer

# === PARAMÈTRES ===
json_path = "JAX_214_007_RGB.json"
dsm_path = "JAX_214_007_RGB_dsm.tif"

# === 1. Charger le JSON et la carte de profondeur ===
with open(json_path) as f:
    data = json.load(f)
rpc = RPCModel(data["rpc"], dict_format="rpcm")

points_2d_raw = np.array(data["keypoints"]["2d_coordinates"])
cols_sampled = points_2d_raw[:, 0].astype(int)
rows_sampled = points_2d_raw[:, 1].astype(int)

with rasterio.open(dsm_path) as dsm:
    alt_map = dsm.read(1)
    height, width = alt_map.shape

# Filtrer les points valides (dans l'image et altitude non-NaN)
valid_mask = (
    (cols_sampled >= 0) & (cols_sampled < width) &
    (rows_sampled >= 0) & (rows_sampled < height) &
    ~np.isnan(alt_map[rows_sampled, cols_sampled])
)
cols_sampled = cols_sampled[valid_mask]
rows_sampled = rows_sampled[valid_mask]
alts_sampled = alt_map[rows_sampled, cols_sampled]

# === 2. Obtenir les points 3D (lon, lat, alt) via RPC ===
lons, lats = rpc.localization(cols_sampled, rows_sampled, alts_sampled)
transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
xs, ys, zs = transformer.transform(lons, lats, alts_sampled)
points_3d = np.vstack([xs, ys, zs]).T.astype(np.float32)

# === 3. Créer les correspondances 2D (cols, rows) ===
points_2d = np.vstack([cols_sampled, rows_sampled]).T.astype(np.float32)

# === 4. Estimer la pose pinhole avec solvePnP ===
focal_est = 30000
K = np.array([[focal_est, 0, width/2],
              [0, focal_est, height/2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, dist_coeffs)
R, _ = cv2.Rodrigues(rvec)
C = -R.T @ tvec

# === 5. Charger le même DSM pour calibration RPC ===
with rasterio.open(dsm_path) as dsm:
    z = dsm.read(1)
    transform = dsm.transform
    crs = dsm.crs

mask = ~np.isnan(z)
rows, cols = np.where(mask)
num_samples = min(5000, len(rows))
idx = np.random.choice(len(rows), size=num_samples, replace=False)
rows_sampled = rows[idx]
cols_sampled = cols[idx]

points_utm = [xy(transform, r, c, offset='center') for r, c in zip(rows_sampled, cols_sampled)]
alts = z[rows_sampled, cols_sampled]

transformer_geo = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
lons, lats = transformer_geo.transform(*zip(*points_utm))
points_3d_world = np.vstack([lons, lats, alts]).T

# === 6. Projection 3D -> 2D avec la pose estimée ===
ecef_transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
ecef_points = np.array([ecef_transformer.transform(*pt) for pt in points_3d_world])
projected_uv = []

for pt in ecef_points:
    vec = pt - C.reshape(-1)
    cam_coords = R.T @ vec
    if cam_coords[2] <= 0:
        projected_uv.append((np.nan, np.nan))
    else:
        u = focal_est * cam_coords[0] / cam_coords[2]
        v = focal_est * cam_coords[1] / cam_coords[2]
        projected_uv.append((u, v))

projected_uv = np.array(projected_uv)
valid = ~np.isnan(projected_uv[:, 0])
points_3d_valid = points_3d_world[valid]
points_2d_valid = projected_uv[valid]

# === 7. Calibrer un RPC artificiel ===
rpc_new = calibrate_rpc(target=points_2d_valid, input_locs=points_3d_valid)

# === 8. Mettre à jour et sauvegarder le JSON d'origine ===
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

final_json = data.copy()
final_json["rpc"] = rpc_dict
output_name = "JAX_214_007_RPC.json"

with open(output_name, "w") as f:
    json.dump(final_json, f, indent=2)

print(f"Pose sauvegardée dans {output_name}")
