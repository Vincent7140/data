import json
import numpy as np
from pyproj import Transformer
import pandas as pd
from rpcm.rpc_model import RPCModel
from calibrate_rpc import calibrate_rpc

# --- Lecture du JSON d'origine ---
with open("JAX_214_007_RGB.json", "r") as f:
    data = json.load(f)

points_2d = np.array(data["keypoints"]["2d_coordinates"])
rpc = RPCModel(data["rpc"])

# --- Localisation : 2D → 3D ---
alt = 10.0
altitudes = np.full((points_2d.shape[0],), alt)
lons, lats = rpc.localization(col=points_2d[:, 0], row=points_2d[:, 1], alt=altitudes)
points_3d = np.vstack([lons, lats, altitudes]).T

# --- Définir une pose satellite simulée ---
lon_center, lat_center = -81.6635, 30.3165
alt_sat = 700000  # en mètres

transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
x_sat, y_sat, z_sat = transformer.transform(lon_center, lat_center, alt_sat)
x_g, y_g, z_g = transformer.transform(lon_center, lat_center, alt)

z_axis = np.array([x_g - x_sat, y_g - y_sat, z_g - z_sat])
z_axis /= np.linalg.norm(z_axis)
north = np.array([0, 0, 1])
x_axis = np.cross(north, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R = np.stack([x_axis, y_axis, z_axis]).T  # matrice de rotation

# --- Projection des points 3D dans l'image simulée ---
points_ecef = np.array([transformer.transform(lon, lat, alt) for lon, lat, alt in points_3d])
f = 1.0  # focale fictive
projected_uv = []

for pt in points_ecef:
    vec = pt - np.array([x_sat, y_sat, z_sat])
    cam_coords = R.T @ vec
    if cam_coords[2] <= 0:
        projected_uv.append((np.nan, np.nan))
    else:
        u = f * cam_coords[0] / cam_coords[2]
        v = f * cam_coords[1] / cam_coords[2]
        projected_uv.append((u, v))

projected_uv = np.array(projected_uv)

# Filtrer les points valides
valid_idx = ~np.isnan(projected_uv[:, 0])
points_3d_valid = points_3d[valid_idx]
points_2d_valid = projected_uv[valid_idx]

# --- Calibration RPC à partir des nouvelles correspondances 3D → 2D ---
rpc_artificial = calibrate_rpc(target=points_2d_valid, input_locs=points_3d_valid)

# --- Sauvegarde du modèle RPC recalibré ---
output_rpc_dict = {
    "row_offset": rpc_artificial.row_offset,
    "col_offset": rpc_artificial.col_offset,
    "lat_offset": rpc_artificial.lat_offset,
    "lon_offset": rpc_artificial.lon_offset,
    "alt_offset": rpc_artificial.alt_offset,
    "row_scale": rpc_artificial.row_scale,
    "col_scale": rpc_artificial.col_scale,
    "lat_scale": rpc_artificial.lat_scale,
    "lon_scale": rpc_artificial.lon_scale,
    "alt_scale": rpc_artificial.alt_scale,
    "row_num": list(rpc_artificial.row_num),
    "row_den": list(rpc_artificial.row_den),
    "col_num": list(rpc_artificial.col_num),
    "col_den": list(rpc_artificial.col_den)
}

final_json = data.copy()
final_json["rpc"] = output_rpc_dict

with open("JAX_214_007_RPC_orbite_pose_simulee.json", "w") as f:
    json.dump(final_json, f, indent=2)
