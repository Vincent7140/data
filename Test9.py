import json
import numpy as np
import cv2
import rasterio
from rpcm.rpc_model import RPCModel
from rpcfit.rpc_fit import calibrate_rpc
from rasterio.transform import xy
from pyproj import Transformer, CRS

# === PARAMÈTRES ===
json_path = "JAX_214_007_RGB.json"
dsm_path = "JAX_214_007_RGB_dsm.tif"
num_points = 5000

# === 1. Charger la pose RPC d'origine et le DSM ===
with open(json_path) as f:
    data = json.load(f)
rpc = RPCModel(data["rpc"], dict_format="rpcm")

with rasterio.open(dsm_path) as dsm:
    alt_map = dsm.read(1)
    height, width = alt_map.shape
    transform = dsm.transform
    dsm_crs = dsm.crs

# === 2. Échantillonnage de points valides ===
rows, cols = np.where(~np.isnan(alt_map))
idx = np.random.choice(len(rows), size=min(num_points, len(rows)), replace=False)
rows_sampled = rows[idx]
cols_sampled = cols[idx]
alts_sampled = alt_map[rows_sampled, cols_sampled]

# === 3. Conversion en coord. géographiques (lon, lat) ===
lons, lats = rpc.localization(cols_sampled, rows_sampled, alts_sampled)

# === 4. Conversion en UTM ===
utm_zone = CRS.from_epsg(32617) if lons[0] > 0 else CRS.from_epsg(32717)  # exemple zone
transformer = Transformer.from_crs("epsg:4326", utm_zone, always_xy=True)
x_utm, y_utm = transformer.transform(lons, lats)
points_3d = np.vstack([x_utm, y_utm, alts_sampled]).T.astype(np.float32)

# === 5. Points image (2D) ===
points_2d = np.vstack([cols_sampled, rows_sampled]).T.astype(np.float32)

# === 6. Estimation de pose pinhole ===
focal_est = 3000
K = np.array([[focal_est, 0, width / 2],
              [0, focal_est, height / 2],
              [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))
success, rvec, tvec = cv2.solvePnP(points_3d, points_2d, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
R, _ = cv2.Rodrigues(rvec)
C = -R.T @ tvec

# === 7. Rééchantillonnage pour calibration ===
rows, cols = np.where(~np.isnan(alt_map))
idx = np.random.choice(len(rows), size=min(num_points, len(rows)), replace=False)
rows_sampled = rows[idx]
cols_sampled = cols[idx]
alts = alt_map[rows_sampled, cols_sampled]

# coord UTM
points_utm = [xy(transform, r, c, offset='center') for r, c in zip(rows_sampled, cols_sampled)]
x_sampled, y_sampled = zip(*points_utm)

# convert back to lat/lon for RPC fitting
transformer_inv = Transformer.from_crs(utm_zone, "epsg:4326", always_xy=True)
lons, lats = transformer_inv.transform(x_sampled, y_sampled)
points_3d_geo = np.vstack([lons, lats, alts]).T

# projection avec pose pinhole
projected_uv = []
for pt in np.vstack([x_sampled, y_sampled, alts]).T:
    vec = pt - C.flatten()
    cam_coords = R.T @ vec
    if cam_coords[2] <= 0:
        projected_uv.append((np.nan, np.nan))
    else:
        u = focal_est * cam_coords[0] / cam_coords[2]
        v = focal_est * cam_coords[1] / cam_coords[2]
        projected_uv.append((u, v))

projected_uv = np.array(projected_uv)
valid = ~np.isnan(projected_uv[:, 0])
points_3d_valid = points_3d_geo[valid]
points_2d_valid = projected_uv[valid]

# === 8. Calibration RPC artificiel ===
rpc_new = calibrate_rpc(target=points_2d_valid, input_locs=points_3d_valid)

# === 9. Sauvegarde ===
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

with open("JAX_214_007_RPC_corrected.json", "w") as f:
    json.dump(final_json, f, indent=2)

print("Nouvelle pose RPC sauvegardée.")
