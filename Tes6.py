# Recharger les dépendances et relancer le traitement suite au reset
import rasterio
import numpy as np
from rasterio.transform import xy
from pyproj import Transformer
from rpcfit.rpc_fit import calibrate_rpc
from rpcm.rpc_model import RPCModel
import json

dsm_path = "JAX_224_CLS.tif"

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

transformer = Transformer.from_crs(crs, "epsg:4326", always_xy=True)
lons, lats = transformer.transform(*zip(*points_utm))
points_3d = np.vstack([lons, lats, alts]).T

# Définir une pose satellite fictive (au-dessus du centre)
lon_center = np.mean(lons)
lat_center = np.mean(lats)
alt_sat = 700000

ecef_transformer = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)
x_sat, y_sat, z_sat = ecef_transformer.transform(lon_center, lat_center, alt_sat)
x_g, y_g, z_g = ecef_transformer.transform(lon_center, lat_center, np.mean(alts))

z_axis = np.array([x_g - x_sat, y_g - y_sat, z_g - z_sat])
z_axis /= np.linalg.norm(z_axis)
north = np.array([0, 0, 1])
x_axis = np.cross(north, z_axis)
x_axis /= np.linalg.norm(x_axis)
y_axis = np.cross(z_axis, x_axis)
R = np.stack([x_axis, y_axis, z_axis]).T

ecef_points = np.array([ecef_transformer.transform(*pt) for pt in points_3d])
f = 1.0
projected_uv = []

for pt in ecef_points:
    vec = pt - np.array([x_sat, y_sat, z_sat])
    cam_coords = R.T @ vec
    if cam_coords[2] <= 0:
        projected_uv.append((np.nan, np.nan))
    else:
        u = f * cam_coords[0] / cam_coords[2]
        v = f * cam_coords[1] / cam_coords[2]
        projected_uv.append((u, v))

projected_uv = np.array(projected_uv)
valid = ~np.isnan(projected_uv[:, 0])
points_3d_valid = points_3d[valid]
points_2d_valid = projected_uv[valid]

rpc_artificial = calibrate_rpc(target=points_2d_valid, input_locs=points_3d_valid)

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

output_path = "/mnt/data/generated_rpc_from_dsm.json"
with open(output_path, "w") as f:
    json.dump(output_rpc_dict, f, indent=2)

output_path
